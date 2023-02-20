# arc-dyn

Provides `ThinArc`, which is an `Arc` that stores the dyn metadata inside the allocation, to give you a thin pointer.

## Example

A thread-safe global future queue (using [`pin_queue`](https://github.com/conradludgate/pin-queue))

```rust
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


// spawn some tasks into the queue
let task1 = ThinArc::pin(Task::new(async { println!("1"); }));
QUEUE.lock().unwrap().push_back(task1).unwrap();

let task2 = ThinArc::pin(Task::new(async { println!("2"); }));
QUEUE.lock().unwrap().push_back(task2).unwrap();


// If tasks are awoken, they get inserted to the back of the queue
impl arc_dyn::pin_queue::ThinWake for DynTask {
    fn wake(task: Pin<ThinArc<DynTask>>) {
        let _ = QUEUE.lock().unwrap().push_back(task);
    }
}


// get tasks from the queue
while let Some(task) = QUEUE.lock().unwrap().pop_front() {
    // use the task as it's own waker!
    let waker = Waker::from(task.clone());
    let mut cx = Context::from_waker(&waker);
    let mut fut = task.as_ref().project_ref().fut.lock();
    let _ = fut.as_mut().poll(&mut cx);
}
```
