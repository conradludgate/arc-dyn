//! Thread-safe reference-counting thin-pointers to dyn data.
//!
//! See the [`Arc<T>`][Arc] documentation for more details.
#![feature(
    unsize,
    layout_for_ptr,
    alloc_layout_extra,
    pointer_byte_offsets,
    slice_ptr_get,
    nonnull_slice_from_raw_parts,
    set_ptr_value,
    ptr_metadata
)]
extern crate alloc;

use core::borrow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::{PhantomData, Unpin};
use core::mem::{self, align_of_val_raw};
use core::ops::Deref;
use core::pin::Pin;
use core::ptr::{self, NonNull};
use core::sync::atomic;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::marker::Unsize;
use std::process::abort;
use std::ptr::{DynMetadata, Pointee};

use alloc::alloc::Layout;
use alloc::boxed::Box;

/// A soft limit on the amount of references that may be made to an `Arc`.
///
/// Going above this limit will abort your program (although not
/// necessarily) at _exactly_ `MAX_REFCOUNT + 1` references.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

macro_rules! acquire {
    ($x:expr) => {
        atomic::fence(Acquire)
    };
}

/// A thread-safe reference-counting pointer. 'Arc' stands for 'Atomically
/// Reference Counted'.
pub struct ThinArc<T: ?Sized>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    ptr: NonNull<ArcInner<T, ()>>,
    phantom: PhantomData<T>,
}

unsafe impl<T: ?Sized + Sync + Send> Send for ThinArc<T> where T: Pointee<Metadata = DynMetadata<T>> {}

unsafe impl<T: ?Sized + Sync + Send> Sync for ThinArc<T> where T: Pointee<Metadata = DynMetadata<T>> {}

// This is repr(C) to future-proof against possible field-reordering, which
// would interfere with otherwise safe [into|from]_raw() of transmutable
// inner types.
#[repr(C)]
struct ArcInner<T: ?Sized, U> {
    header: ArcHeader<DynMetadata<T>>,
    data: U,
}
struct ArcHeader<H> {
    strong: atomic::AtomicUsize,
    metadata: H,
}

impl<T: ?Sized> ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    /// Constructs a new `Arc<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// ```

    #[inline]

    pub fn new<U: Unsize<T>>(data: U) -> ThinArc<T> {
        // Start the weak pointer count as 1 which is the weak pointer that's
        // held by all the strong pointers (kinda), see std/rc.rs for more info
        let x: Box<_> = Box::new(ArcInner {
            header: ArcHeader {
                strong: atomic::AtomicUsize::new(1),
                metadata: ptr::metadata(&data as &T),
            },
            data,
        });
        Self {
            ptr: NonNull::from(Box::leak(x)).cast(),
            phantom: PhantomData,
        }
    }

    /// Constructs a new `Pin<Arc<T>>`. If `T` does not implement `Unpin`, then
    /// `data` will be pinned in memory and unable to be moved.
    #[must_use]
    pub fn pin<U: Unsize<T>>(data: U) -> Pin<ThinArc<T>> {
        unsafe { Pin::new_unchecked(ThinArc::new(data)) }
    }
}

impl<T: ?Sized> ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    /// Consumes the `ThinArc`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `ThinArc` using
    /// [`ThinArc::from_raw`].
    #[must_use = "losing the pointer will leak memory"]
    pub fn into_raw(this: Self) -> *const T {
        let ptr = Self::as_ptr(&this);
        mem::forget(this);
        ptr
    }

    /// Provides a raw pointer to the data.
    ///
    /// The counts are not affected in any way and the `ThinArc` is not consumed. The pointer is valid for
    /// as long as there are strong counts in the `ThinArc`.
    #[must_use]
    pub fn as_ptr(this: &Self) -> *const T {
        let metadata = unsafe { *ptr::addr_of!((*this.ptr.as_ptr()).header.metadata) };
        unsafe {
            let offset = data_offset_align::<T>(metadata.align_of());

            // Reverse the offset to find the original ArcInner.
            let data_ptr = (this.ptr.as_ptr() as *const _ as *const ()).byte_add(offset);

            ptr::from_raw_parts(data_ptr, metadata)
        }
    }

    /// Constructs an `ThinArc<T>` from a raw pointer.
    ///
    /// The raw pointer must have been previously returned by a call to
    /// [`ThinArc<U>::into_raw`][into_raw] where `U` must have the same size and
    /// alignment as `T`. This is trivially true if `U` is `T`.
    /// Note that if `U` is not `T` but has the same size and alignment, this is
    /// basically like transmuting references of different types. See
    /// [`mem::transmute`][transmute] for more information on what
    /// restrictions apply in this case.
    ///
    /// The user of `from_raw` has to make sure a specific value of `T` is only
    /// dropped once.
    ///
    /// This function is unsafe because improper use may lead to memory unsafety,
    /// even if the returned `ThinArc<T>` is never accessed.
    ///
    /// [into_raw]: ThinArc::into_raw
    /// [transmute]: core::mem::transmute
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        unsafe {
            let offset = data_offset(ptr);

            // Reverse the offset to find the original ArcInner.
            let arc_ptr = ptr.byte_sub(offset) as *mut ArcInner<T, ()>;

            Self {
                ptr: NonNull::new_unchecked(arc_ptr),
                phantom: PhantomData,
            }
        }
    }

    /// Gets the number of strong (`ThinArc`) pointers to this allocation.
    ///
    /// # Safety
    ///
    /// This method by itself is safe, but using it correctly requires extra care.
    /// Another thread can change the strong count at any time,
    /// including potentially between calling this method and acting on the result.
    #[inline]
    #[must_use]
    pub fn strong_count(this: &Self) -> usize {
        this.inner().header.strong.load(Acquire)
    }

    #[inline]
    fn inner(&self) -> &ArcInner<T, ()> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `ArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        // contents.
        unsafe { self.ptr.as_ref() }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        // Destroy the data at this time, even though we must not free the box
        // allocation itself.
        let metadata = unsafe { *ptr::addr_of!((*self.ptr.as_ptr()).header.metadata) };
        unsafe {
            let offset = data_offset_align::<T>(metadata.align_of());

            // Reverse the offset to find the original ArcInner.
            let data_ptr = (self.ptr.as_ptr() as *mut _ as *mut ()).byte_add(offset);

            ptr::drop_in_place(ptr::from_raw_parts_mut::<T>(data_ptr, metadata));
        }
        unsafe {
            alloc::alloc::dealloc(
                self.ptr.as_ptr().cast(),
                Layout::for_value_raw(self.ptr.as_ptr())
                    .extend(metadata.layout())
                    .unwrap()
                    .0
                    .pad_to_align(),
            )
        }
    }

    /// Returns `true` if the two `ThinArc`s point to the same allocation in a vein similar to
    /// [`ptr::eq`]. See [that function][`ptr::eq`] for caveats when comparing `dyn Trait` pointers.
    ///
    /// [`ptr::eq`]: core::ptr::eq "ptr::eq"
    #[inline]
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T: ?Sized> Deref for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        let metadata = unsafe { self.ptr.as_ref().header.metadata };
        unsafe {
            let offset = data_offset_align::<T>(metadata.align_of());
            let data_ptr = (self.ptr.as_ptr() as *const _ as *const ()).byte_add(offset);
            &*ptr::from_raw_parts(data_ptr, metadata)
        }
    }
}

impl<T: ?Sized> Drop for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    /// Drops the `ThinArc`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count reaches zero we `drop` the inner value.
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        if self.inner().header.strong.fetch_sub(1, Release) != 1 {
            return;
        }

        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data.  Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` fence. This
        // means that use of the data happens before decreasing the reference
        // count, which happens before this fence, which happens before the
        // deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // > It is important to enforce any possible access to the object in one
        // > thread (through an existing reference) to *happen before* deleting
        // > the object in a different thread. This is achieved by a "release"
        // > operation after dropping a reference (any access to the object
        // > through this reference must obviously happened before), and an
        // > "acquire" operation before deleting the object.
        //
        // In particular, while the contents of an Arc are usually immutable, it's
        // possible to have interior writes to something like a Mutex<T>. Since a
        // Mutex is not acquired when it is deleted, we can't rely on its
        // synchronization logic to make writes in thread A visible to a destructor
        // running in thread B.
        //
        // Also note that the Acquire fence here could probably be replaced with an
        // Acquire load, which could improve performance in highly-contended
        // situations. See [2].
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        // [2]: (https://github.com/rust-lang/rust/pull/41714)
        acquire!(self.inner().strong);

        unsafe {
            self.drop_slow();
        }
    }
}

impl<T: ?Sized> Clone for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    #[inline]
    fn clone(&self) -> ThinArc<T> {
        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        let old_size = self.inner().header.strong.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone is `mem::forget`ing
        // Arcs. If we don't do this the count can overflow and users will use-after free. This
        // branch will never be taken in any realistic program. We abort because such a program is
        // incredibly degenerate, and we don't care to support it.
        //
        // This check is not 100% water-proof: we error when the refcount grows beyond `isize::MAX`.
        // But we do that check *after* having done the increment, so there is a chance here that
        // the worst already happened and we actually do overflow the `usize` counter. However, that
        // requires the counter to grow from `isize::MAX` to `usize::MAX` between the increment
        // above and the `abort` below, which seems exceedingly unlikely.
        if old_size > MAX_REFCOUNT {
            abort();
        }

        ThinArc {
            ptr: self.ptr,
            phantom: PhantomData,
        }
    }
}

impl<T: ?Sized + PartialEq> PartialEq for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    #[inline]
    fn eq(&self, other: &ThinArc<T>) -> bool {
        **self == **other
    }
}

impl<T: ?Sized + PartialOrd> PartialOrd for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    /// Partial comparison for two `ThinArc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    fn partial_cmp(&self, other: &ThinArc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: ?Sized + Ord> Ord for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    /// Comparison for two `ThinArc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    fn cmp(&self, other: &ThinArc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + Eq> Eq for ThinArc<T> where T: Pointee<Metadata = DynMetadata<T>> {}

impl<T: ?Sized + fmt::Display> fmt::Display for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized> fmt::Pointer for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

impl<T: ?Sized + Hash> Hash for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: ?Sized> borrow::Borrow<T> for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: ?Sized> AsRef<T> for ThinArc<T>
where
    T: Pointee<Metadata = DynMetadata<T>>,
{
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized> Unpin for ThinArc<T> where T: Pointee<Metadata = DynMetadata<T>> {}

/// Get the offset within an `ArcInner` for the payload behind a pointer.
///
/// # Safety
///
/// The pointer must point to (and have valid metadata for) a previously
/// valid instance of T, but the T is allowed to be dropped.
unsafe fn data_offset<T: ?Sized>(ptr: *const T) -> usize {
    // Align the unsized value to the end of the ArcInner.
    // Because RcBox is repr(C), it will always be the last field in memory.
    // SAFETY: since the only unsized types possible are slices, trait objects,
    // and extern types, the input safety requirement is currently enough to
    // satisfy the requirements of align_of_val_raw; this is an implementation
    // detail of the language that must not be relied upon outside of std.
    unsafe { data_offset_align::<T>(align_of_val_raw(ptr)) }
}

#[inline]
fn data_offset_align<T: ?Sized>(align: usize) -> usize {
    let layout = Layout::new::<ArcHeader<DynMetadata<T>>>();
    layout.size() + layout.padding_needed_for(align)
}

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use crate::ThinArc;

    #[test]
    fn clone_works() {
        let x: ThinArc<dyn Display> = ThinArc::new(1usize);
        let y = x.clone();

        drop(x);
        assert_eq!(y.to_string(), "1");
    }

    #[test]
    fn many_types() {
        let mut x: ThinArc<dyn Display>;

        x = ThinArc::new(1usize);
        assert_eq!(x.to_string(), "1");

        x = ThinArc::new("hello");
        assert_eq!(x.to_string(), "hello");

        x = ThinArc::new(true);
        assert_eq!(x.to_string(), "true");
    }

    #[test]
    fn single_width() {
        assert_eq!(
            std::mem::size_of::<ThinArc<dyn Display>>(),
            std::mem::size_of::<*const ()>()
        );
    }

    #[test]
    fn handles_drop() {
        struct Count<'a>(&'a mut usize);
        impl Drop for Count<'_> {
            fn drop(&mut self) {
                *self.0 += 1;
            }
        }
        let mut drop_count = 0;
        {
            let x: ThinArc<dyn Send> = ThinArc::new(Count(&mut drop_count));
            let _y = x.clone();
            let _z = x.clone();
        }
        assert_eq!(drop_count, 1);
    }
}
