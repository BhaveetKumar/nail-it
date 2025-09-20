use crossbeam::scope;
use crossbeam_channel as channel;
use rand::{thread_rng, Rng};

pub fn scoped_sum(numbers: &[i64], chunks: usize) -> i64 {
    assert!(chunks > 0);
    let chunk_size = (numbers.len() + chunks - 1) / chunks;
    let (tx, rx) = channel::unbounded::<i64>();

    scope(|s| {
        for (i, chunk) in numbers.chunks(chunk_size).enumerate() {
            let tx = tx.clone();
            s.spawn(move |_| {
                let sum = chunk.iter().copied().sum::<i64>();
                tx.send(sum).unwrap();
            });
            if i + 1 == chunks { break; }
        }
    })
    .unwrap();
    drop(tx);

    rx.iter().sum()
}

pub fn worker_pool(num_workers: usize, jobs: usize) -> usize {
    // Fan-out (workers) and fan-in (results) with channels
    let (job_tx, job_rx) = channel::bounded::<usize>(num_workers);
    let (res_tx, res_rx) = channel::unbounded::<usize>();

    scope(|s| {
        for _ in 0..num_workers {
            let rx = job_rx.clone();
            let tx = res_tx.clone();
            s.spawn(move |_| {
                for j in rx.iter() {
                    // do some pretend work
                    let mut rng = thread_rng();
                    let delay: u64 = rng.gen_range(0..2);
                    if delay == 1 { std::thread::yield_now(); }
                    tx.send(j * j).unwrap();
                }
            });
        }

        for j in 0..jobs {
            job_tx.send(j).unwrap();
        }
        drop(job_tx);
    })
    .unwrap();
    drop(res_tx);

    res_rx.iter().count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_sum_matches_sequential() {
        let data: Vec<i64> = (1..=10_000).collect();
        let seq = data.iter().sum::<i64>();
        let par = scoped_sum(&data, 4);
        assert_eq!(seq, par);
    }

    #[test]
    fn worker_pool_processes_all_jobs() {
        let count = worker_pool(4, 1000);
        assert_eq!(count, 1000);
    }
}
