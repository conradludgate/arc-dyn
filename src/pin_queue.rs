//! This module provides [`PinQueue`](pin_queue::PinQueue) support.
//!
//! # Example
//!
//! A thread-safe global future queue.
//!
/*! ```
use std::cell::UnsafeCell;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Waker};
use std::sync::Mutex;
use once_cell::sync::Lazy;
use arc_dyn::ThinArc;

// aliases
type DynTask = Task<dyn Future<Output = ()> + Send + Sync + 'static>;
type QueueTypes = dyn pin_queue::Types<
    Id = pin_queue::id::Checked,
    Key = Key,
    Pointer = ThinArc<DynTask>,
>;
type PinQueue = pin_queue::PinQueue<QueueTypes>;

// our Task type that stores the intrusive pointers and our futures
pin_project_lite::pin_project! (
    struct Task<F: ?Sized> {
        #[pin]
        intrusive: pin_queue::Intrusive<QueueTypes>,
        #[pin]
        fut: pin_lock::PinLock<F>,
    }
);

impl<F> Task<F> {
    pub fn new(fut: F) -> Self {
        Self {
            intrusive: pin_queue::Intrusive::new(),
            fut: pin_lock::PinLock::new(fut),
        }
    }
}

struct Key;
impl pin_queue::GetIntrusive<QueueTypes> for Key {
    fn get_intrusive(p: Pin<&DynTask>) -> Pin<&pin_queue::Intrusive<QueueTypes>> {
        p.project_ref().intrusive
    }
}

// global queue
static QUEUE: Lazy<Mutex<PinQueue>> = Lazy::new(|| {
    Mutex::new(PinQueue::new(pin_queue::id::Checked::new()))
});

// spawn()
let task1 = ThinArc::pin(Task::new(async { println!("1"); }));
QUEUE.lock().unwrap().push_back(task1).unwrap();

let task2 = ThinArc::pin(Task::new(async { println!("2"); }));
QUEUE.lock().unwrap().push_back(task2).unwrap();

// waker
impl arc_dyn::pin_queue::ThinWake for DynTask {
    fn wake(task: Pin<ThinArc<DynTask>>) {
        let _ = QUEUE.lock().unwrap().push_back(task);
    }
}

// worker
while let Some(task) = QUEUE.lock().unwrap().pop_front() {
    let waker = Waker::from(task.clone());
    let mut cx = Context::from_waker(&waker);
    let mut fut = task.as_ref().project_ref().fut.lock();
    let _ = fut.as_mut().poll(&mut cx);
}
``` */

use std::{
    mem::ManuallyDrop,
    pin::Pin,
    ptr::NonNull,
    task::{RawWaker, RawWakerVTable, Waker},
};

use pin_queue::SharedPointer;

use crate::{header::WithOpaqueHeader, ThinArc};

// Safety: this is an aliasable shared pointer
unsafe impl<T: ?Sized + 'static> SharedPointer for ThinArc<T> {}

pub trait ThinWake {
    fn wake(thin: Pin<ThinArc<Self>>);
}

impl<W: ThinWake + ?Sized + Send + Sync + 'static> From<Pin<ThinArc<W>>> for Waker {
    /// Use a `Wake`-able type as a `Waker`.
    ///
    /// No heap allocations or atomic operations are used for this conversion.
    fn from(waker: Pin<ThinArc<W>>) -> Waker {
        // SAFETY: This is safe because raw_waker safely constructs
        // a RawWaker from Arc<W>.
        unsafe { Waker::from_raw(raw_waker(waker)) }
    }
}

impl<W: ThinWake + ?Sized + Send + Sync + 'static> From<Pin<ThinArc<W>>> for RawWaker {
    /// Use a `Wake`-able type as a `RawWaker`.
    ///
    /// No heap allocations or atomic operations are used for this conversion.
    fn from(waker: Pin<ThinArc<W>>) -> RawWaker {
        raw_waker(waker)
    }
}

// NB: This private function for constructing a RawWaker is used, rather than
// inlining this into the `From<ThinArc<W>> for RawWaker` impl, to ensure that
// the safety of `From<ThinArc<W>> for Waker` does not depend on the correct
// trait dispatch - instead both impls call this function directly and
// explicitly.
#[inline(always)]
fn raw_waker<W: ThinWake + ?Sized + Send + Sync + 'static>(waker: Pin<ThinArc<W>>) -> RawWaker {
    unsafe fn from_thin<W: ThinWake + ?Sized + Send + Sync + 'static>(
        waker: *const (),
    ) -> ManuallyDrop<Pin<ThinArc<W>>> {
        ManuallyDrop::new(unsafe {
            Pin::new_unchecked(ThinArc {
                ptr: WithOpaqueHeader(NonNull::new_unchecked(waker.cast_mut().cast())),
                _marker: std::marker::PhantomData,
            })
        })
    }

    // Increment the reference count of the arc to clone it.
    unsafe fn clone_waker<W: ThinWake + ?Sized + Send + Sync + 'static>(
        waker: *const (),
    ) -> RawWaker {
        let arc = from_thin::<W>(waker);
        let _arc = arc.clone();
        RawWaker::new(
            waker as *const (),
            &RawWakerVTable::new(
                clone_waker::<W>,
                wake::<W>,
                wake_by_ref::<W>,
                drop_waker::<W>,
            ),
        )
    }

    // Wake by value, moving the Arc into the Wake::wake function
    unsafe fn wake<W: ThinWake + ?Sized + Send + Sync + 'static>(waker: *const ()) {
        let waker = unsafe { ManuallyDrop::into_inner(from_thin::<W>(waker)) };
        <W as ThinWake>::wake(waker);
    }

    // Wake by reference, wrap the waker in ManuallyDrop to avoid dropping it
    unsafe fn wake_by_ref<W: ThinWake + ?Sized + Send + Sync + 'static>(waker: *const ()) {
        let waker = unsafe { from_thin(waker) };
        <W as ThinWake>::wake(Pin::<ThinArc<W>>::clone(&waker));
    }

    // Decrement the reference count of the ThinArc on drop
    unsafe fn drop_waker<W: ThinWake + ?Sized + Send + Sync + 'static>(waker: *const ()) {
        unsafe { ManuallyDrop::into_inner(from_thin::<W>(waker)) };
    }

    RawWaker::new(
        unsafe {
            ManuallyDrop::new(Pin::into_inner_unchecked(waker))
                .ptr
                .0
                .as_ptr()
                .cast()
        },
        &RawWakerVTable::new(
            clone_waker::<W>,
            wake::<W>,
            wake_by_ref::<W>,
            drop_waker::<W>,
        ),
    )
}

#[cfg(test)]
mod tests {
    use crate::ThinArc;
    use once_cell::sync::Lazy;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;
    use std::task::{Context, Waker};

    // aliases
    type DynNode = Node<dyn Future<Output = ()> + Send + Sync + 'static>;
    type PinQueueTypes =
        dyn pin_queue::Types<Id = pin_queue::id::Checked, Key = Key, Pointer = ThinArc<DynNode>>;
    type PinQueue = pin_queue::PinQueue<PinQueueTypes>;

    // our node type that stores the intrusive pointers and our value
    pin_project_lite::pin_project!(
        struct Node<V: ?Sized> {
            #[pin]
            intrusive: pin_queue::Intrusive<PinQueueTypes>,
            #[pin]
            value: pin_lock::PinLock<V>,
        }
    );
    impl<V> Node<V> {
        pub fn new(value: V) -> Self {
            Self {
                intrusive: pin_queue::Intrusive::new(),
                value: pin_lock::PinLock::new(value),
            }
        }
    }

    struct Key;
    impl pin_queue::GetIntrusive<PinQueueTypes> for Key {
        fn get_intrusive(p: Pin<&DynNode>) -> Pin<&pin_queue::Intrusive<PinQueueTypes>> {
            p.project_ref().intrusive
        }
    }

    // global queue
    static QUEUE: Lazy<Mutex<PinQueue>> =
        Lazy::new(|| Mutex::new(PinQueue::new(pin_queue::id::Checked::new())));

    #[test]
    fn test() {
        // spawn()
        let task1 = ThinArc::pin(Node::new(async {
            println!("1");
        }));
        QUEUE.lock().unwrap().push_back(task1).unwrap();

        let task2 = ThinArc::pin(Node::new(async {
            println!("2");
        }));
        QUEUE.lock().unwrap().push_back(task2).unwrap();

        // waker
        impl crate::pin_queue::ThinWake for DynNode {
            fn wake(task: Pin<ThinArc<DynNode>>) {
                let _ = QUEUE.lock().unwrap().push_back(task);
            }
        }

        // worker
        while let Some(task) = QUEUE.lock().unwrap().pop_front() {
            let waker = Waker::from(task.clone());
            let mut cx = Context::from_waker(&waker);
            let mut fut = task.as_ref().project_ref().value.lock();
            let _ = fut.as_mut().poll(&mut cx);
        }
    }
}
