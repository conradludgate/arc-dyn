//! Thread-safe reference-counting thin-pointers to dyn data.
//!
//! See the [`ThinArc<T>`][ThinArc] documentation for more details.
#![feature(unsize, layout_for_ptr, alloc_layout_extra, ptr_metadata)]

use core::borrow;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::{PhantomData, Unpin};
use core::mem;
use core::ops::Deref;
use core::pin::Pin;
use core::ptr::{self, NonNull};
use core::sync::atomic;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::marker::Unsize;
use std::process::abort;
use std::ptr::Pointee;

use header::{WithHeader, WithOpaqueHeader};

mod header;
pub mod pin_queue;

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
pub struct ThinArc<T: ?Sized> {
    // This is essentially `WithHeader<ArcHeader<<T as Pointee>::Metadata>>`,
    // but that would be invariant in `T`, and we want covariance.
    pub(crate) ptr: WithOpaqueHeader,
    _marker: PhantomData<T>,
}

unsafe impl<T: ?Sized + Sync + Send> Send for ThinArc<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for ThinArc<T> {}

struct ArcHeader<H> {
    strong: atomic::AtomicUsize,
    metadata: H,
}

impl<T: ?Sized> ThinArc<T> {
    /// Constructs a new `ThinArc<T>`
    #[inline]
    pub fn new<U: Unsize<T>>(value: U) -> ThinArc<T> {
        let header = ArcHeader {
            metadata: ptr::metadata(&value as &T),
            strong: atomic::AtomicUsize::new(1),
        };
        let ptr = WithOpaqueHeader::new(header, value);
        ThinArc {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Constructs a new `Pin<ThinArc<T>>`. If `T` does not implement `Unpin`, then
    /// `value` will be pinned in memory and unable to be moved.
    #[must_use]
    pub fn pin<U: Unsize<T>>(value: U) -> Pin<ThinArc<T>> {
        unsafe { Pin::new_unchecked(ThinArc::new(value)) }
    }
}

impl<T: ?Sized> ThinArc<T> {
    fn meta(&self) -> <T as Pointee>::Metadata {
        //  Safety:
        //  -   NonNull and valid.
        unsafe { *ptr::addr_of!((*self.with_header().header()).metadata) }
    }

    fn data(&self) -> *mut u8 {
        self.with_header().value()
    }

    fn with_header(&self) -> &WithHeader<ArcHeader<<T as Pointee>::Metadata>> {
        // SAFETY: both types are transparent to `NonNull<u8>`
        unsafe { &*((&self.ptr) as *const WithOpaqueHeader as *const WithHeader<_>) }
    }

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
        let value = this.data();
        let metadata = this.meta();
        ptr::from_raw_parts(value as *const (), metadata)
    }

    /// Constructs an `ThinArc<T>` from a raw pointer.
    ///
    /// # Safety
    /// Must be produced from a [`ThinArc::into_raw`] call
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        Self {
            ptr: WithOpaqueHeader(NonNull::new_unchecked(ptr.cast_mut().cast())),
            _marker: PhantomData,
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
        this.header().strong.load(Acquire)
    }

    #[inline]
    fn header(&self) -> &ArcHeader<<T as Pointee>::Metadata> {
        unsafe { &*self.with_header().header() }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        let ptr = Self::as_ptr(self);
        self.with_header().drop::<T>(ptr.cast_mut());
    }

    /// Returns `true` if the two `ThinArc`s point to the same allocation in a vein similar to
    /// [`ptr::eq`]. See [that function][`ptr::eq`] for caveats when comparing `dyn Trait` pointers.
    ///
    /// [`ptr::eq`]: core::ptr::eq "ptr::eq"
    #[inline]
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr.0.as_ptr() == other.ptr.0.as_ptr()
    }
}

impl<T: ?Sized> Deref for ThinArc<T> {
    type Target = T;

    fn deref(&self) -> &T {
        let value = self.data();
        let metadata = self.meta();
        let pointer = ptr::from_raw_parts(value as *const (), metadata);
        unsafe { &*pointer }
    }
}

impl<T: ?Sized> Drop for ThinArc<T> {
    /// Drops the `ThinArc`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count reaches zero we `drop` the inner value.
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        if self.header().strong.fetch_sub(1, Release) != 1 {
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

impl<T: ?Sized> Clone for ThinArc<T> {
    #[inline]
    fn clone(&self) -> Self {
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
        let old_size = self.header().strong.fetch_add(1, Relaxed);

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
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized + PartialEq> PartialEq for ThinArc<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: ?Sized + PartialOrd> PartialOrd for ThinArc<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: ?Sized + Ord> Ord for ThinArc<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + Eq> Eq for ThinArc<T> {}

impl<T: ?Sized + fmt::Display> fmt::Display for ThinArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for ThinArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized> fmt::Pointer for ThinArc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

impl<T: ?Sized + Hash> Hash for ThinArc<T> {
    fn hash<H1: Hasher>(&self, state: &mut H1) {
        (**self).hash(state)
    }
}

impl<T: ?Sized> borrow::Borrow<T> for ThinArc<T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: ?Sized> AsRef<T> for ThinArc<T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized> Unpin for ThinArc<T> {}

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
