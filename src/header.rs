//! headers taken from <https://doc.rust-lang.org/src/alloc/boxed/thin.rs.html>.

use std::{
    alloc::{self, Layout, LayoutError},
    marker::PhantomData,
    mem,
    ptr::{self, NonNull},
};

/// A pointer to type-erased data, guaranteed to either be:
/// 1. `NonNull::dangling()`, in the case where both the pointee (`T`) and
///    metadata (`H`) are ZSTs.
/// 2. A pointer to a valid `T` that has a header `H` directly before the
///    pointed-to location.
#[repr(transparent)]
pub(crate) struct WithHeader<H>(NonNull<u8>, PhantomData<H>);

/// An opaque representation of `WithHeader<H>` to avoid the
/// projection invariance of `<T as Pointee>::Metadata`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(crate) struct WithOpaqueHeader(pub(crate) NonNull<u8>);

impl WithOpaqueHeader {
    pub(crate) fn new<H, T>(header: H, value: T) -> Self {
        let ptr = WithHeader::new(header, value);
        Self(ptr.0)
    }
}

impl<H> WithHeader<H> {
    fn new<T>(header: H, value: T) -> WithHeader<H> {
        let value_layout = Layout::new::<T>();
        let Ok((layout, value_offset)) = Self::alloc_layout(value_layout) else {
            // We pass an empty layout here because we do not know which layout caused the
            // arithmetic overflow in `Layout::extend` and `handle_alloc_error` takes `Layout` as
            // its argument rather than `Result<Layout, LayoutError>`, also this function has been
            // stable since 1.28 ._.
            //
            // On the other hand, look at this gorgeous turbofish!
            alloc::handle_alloc_error(Layout::new::<()>());
        };

        unsafe {
            // Note: It's UB to pass a layout with a zero size to `alloc::alloc`, so
            // we use `layout.dangling()` for this case, which should have a valid
            // alignment for both `T` and `H`.
            let ptr = if layout.size() == 0 {
                // Some paranoia checking, mostly so that the ThinBox tests are
                // more able to catch issues.
                debug_assert!(
                    value_offset == 0 && mem::size_of::<T>() == 0 && mem::size_of::<H>() == 0
                );
                layout.dangling()
            } else {
                let ptr = alloc::alloc(layout);
                if ptr.is_null() {
                    alloc::handle_alloc_error(layout);
                }
                // Safety:
                // - The size is at least `aligned_header_size`.
                let ptr = ptr.add(value_offset) as *mut _;

                NonNull::new_unchecked(ptr)
            };

            let result = WithHeader(ptr, PhantomData);
            ptr::write(result.header(), header);
            ptr::write(result.value().cast(), value);

            result
        }
    }

    // Safety:
    // - Assumes that either `value` can be dereferenced, or is the
    //   `NonNull::dangling()` we use when both `T` and `H` are ZSTs.
    pub(crate) unsafe fn drop<T: ?Sized>(&self, value: *mut T) {
        unsafe {
            let value_layout = Layout::for_value_raw(value);
            // SAFETY: Layout must have been computable if we're in drop
            let (layout, value_offset) = Self::alloc_layout(value_layout).unwrap_unchecked();

            // We only drop the value because the Pointee trait requires that the metadata is copy
            // aka trivially droppable.
            ptr::drop_in_place::<T>(value);

            // Note: Don't deallocate if the layout size is zero, because the pointer
            // didn't come from the allocator.
            if layout.size() != 0 {
                alloc::dealloc(self.0.as_ptr().sub(value_offset), layout);
            } else {
                debug_assert!(
                    value_offset == 0 && mem::size_of::<H>() == 0 && value_layout.size() == 0
                );
            }
        }
    }

    pub(crate) fn header(&self) -> *mut H {
        //  Safety:
        //  - At least `size_of::<H>()` bytes are allocated ahead of the pointer.
        //  - We know that H will be aligned because the middle pointer is aligned to the greater
        //    of the alignment of the header and the data and the header size includes the padding
        //    needed to align the header. Subtracting the header size from the aligned data pointer
        //    will always result in an aligned header pointer, it just may not point to the
        //    beginning of the allocation.
        unsafe { self.0.as_ptr().sub(Self::header_size()) as *mut H }
    }

    pub(crate) fn value(&self) -> *mut u8 {
        self.0.as_ptr()
    }

    pub(crate) const fn header_size() -> usize {
        mem::size_of::<H>()
    }

    pub(crate) fn alloc_layout(value_layout: Layout) -> Result<(Layout, usize), LayoutError> {
        Layout::new::<H>().extend(value_layout)
    }
}
