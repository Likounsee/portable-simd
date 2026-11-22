//! Generic-length array strategy.

// Adapted from proptest's array code
// Copyright 2017 Jason Lingle

use core::{marker::PhantomData, mem::MaybeUninit};
use proptest::{
    strategy::{NewTree, Strategy, ValueTree},
    test_runner::TestRunner,
};

#[must_use = "strategies do nothing unless used"]
#[derive(Clone, Copy, Debug)]
pub struct UniformArrayStrategy<S, T> {
    strategy: S,
    _marker: PhantomData<T>,
}

impl<S, T> UniformArrayStrategy<S, T> {
    pub const fn new(strategy: S) -> Self {
        Self {
            strategy,
            _marker: PhantomData,
        }
    }
}

pub struct ArrayValueTree<T> {
    tree: T,
    shrinker: usize,
    last_shrinker: Option<usize>,
}

// Owns the initialized prefix of an array under construction. It is deliberately
// a raw-pointer guard so that both an ordinary error return and unwinding clean up
// every value already written into the MaybeUninit array.
struct ArrayInitGuard<T> {
    ptr: *mut T,
    initialized: usize,
}

impl<T> Drop for ArrayInitGuard<T> {
    fn drop(&mut self) {
        for index in 0..self.initialized {
            // SAFETY: the constructor increments initialized only after writing
            // a valid T into this slot; all earlier slots are therefore initialized.
            unsafe { self.ptr.add(index).drop_in_place() };
        }
    }
}

fn try_init_array<T, E, const N: usize>(
    mut init: impl FnMut(usize) -> Result<T, E>,
) -> Result<[T; N], E> {
    let mut array = MaybeUninit::<[T; N]>::uninit();
    let ptr = array.as_mut_ptr().cast::<T>();
    let mut guard = ArrayInitGuard {
        ptr,
        initialized: 0,
    };

    while guard.initialized < N {
        let value = init(guard.initialized)?;
        // SAFETY: initialized is strictly less than N, and this slot has not
        // previously been written. The guard tracks the value immediately after.
        unsafe { ptr.add(guard.initialized).write(value) };
        guard.initialized += 1;
    }

    // Transfer ownership of all N initialized values to the returned array.
    guard.initialized = 0;
    // SAFETY: every element was initialized by the loop above.
    Ok(unsafe { array.assume_init() })
}

impl<T, S, const LANES: usize> Strategy for UniformArrayStrategy<S, [T; LANES]>
where
    T: core::fmt::Debug,
    S: Strategy<Value = T>,
{
    type Tree = ArrayValueTree<[S::Tree; LANES]>;
    type Value = [T; LANES];

    fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
        let tree = try_init_array(|_| self.strategy.new_tree(runner))?;
        Ok(ArrayValueTree {
            tree,
            shrinker: 0,
            last_shrinker: None,
        })
    }
}

impl<T: ValueTree, const LANES: usize> ValueTree for ArrayValueTree<[T; LANES]> {
    type Value = [T::Value; LANES];

    fn current(&self) -> Self::Value {
        match try_init_array(|index| {
            Ok::<T::Value, core::convert::Infallible>(self.tree[index].current())
        }) {
            Ok(value) => value,
            Err(never) => match never {},
        }
    }

    fn simplify(&mut self) -> bool {
        while self.shrinker < LANES {
            if self.tree[self.shrinker].simplify() {
                self.last_shrinker = Some(self.shrinker);
                return true;
            } else {
                self.shrinker += 1;
            }
        }

        false
    }

    fn complicate(&mut self) -> bool {
        if let Some(shrinker) = self.last_shrinker {
            self.shrinker = shrinker;
            if self.tree[shrinker].complicate() {
                true
            } else {
                self.last_shrinker = None;
                false
            }
        } else {
            false
        }
    }
}


#[cfg(test)]
mod tests {
    use super::try_init_array;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn initialized_prefix_is_dropped_when_initialization_returns_error() {
        let drops = Arc::new(AtomicUsize::new(0));
        let result = try_init_array::<_, (), 4>(|index| {
            if index == 2 {
                Err(())
            } else {
                Ok(DropCounter(Arc::clone(&drops)))
            }
        });

        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn initialized_prefix_is_dropped_when_initialization_panics() {
        let drops = Arc::new(AtomicUsize::new(0));
        let result = std::panic::catch_unwind({
            let drops = Arc::clone(&drops);
            move || {
                let _ = try_init_array::<_, (), 4>(|index| {
                    if index == 2 {
                        panic!("simulated strategy panic");
                    }
                    Ok(DropCounter(Arc::clone(&drops)))
                });
            }
        });

        assert!(result.is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn successful_initialization_transfers_all_elements_to_the_array() {
        let drops = Arc::new(AtomicUsize::new(0));
        let values = try_init_array::<_, (), 3>(|_| Ok(DropCounter(Arc::clone(&drops))))
            .expect("initialization succeeds");

        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(values);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
    }
}
