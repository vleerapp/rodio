use crate::buffer::SamplesBuffer;
use crate::math::nz;
use crate::mixer;
use crate::source::Source;

#[test]
fn basic() {
    let (tx, mut rx) = mixer::mixer(nz!(1), nz!(48000));

    tx.add(SamplesBuffer::new(
        nz!(1),
        nz!(48000),
        vec![10.0, -10.0, 10.0, -10.0],
    ));
    tx.add(SamplesBuffer::new(
        nz!(1),
        nz!(48000),
        vec![5.0, 5.0, 5.0, 5.0],
    ));

    assert_eq!(rx.channels(), nz!(1));
    assert_eq!(rx.sample_rate().get(), 48000);
    assert_eq!(rx.next(), Some(15.0));
    assert_eq!(rx.next(), Some(-5.0));
    assert_eq!(rx.next(), Some(15.0));
    assert_eq!(rx.next(), Some(-5.0));
    assert_eq!(rx.next(), None);
}

#[test]
fn channels_conv() {
    let (tx, mut rx) = mixer::mixer(nz!(2), nz!(48000));

    tx.add(SamplesBuffer::new(
        nz!(1),
        nz!(48000),
        vec![10.0, -10.0, 10.0, -10.0],
    ));
    tx.add(SamplesBuffer::new(
        nz!(1),
        nz!(48000),
        vec![5.0, 5.0, 5.0, 5.0],
    ));

    assert_eq!(rx.channels(), nz!(2));
    assert_eq!(rx.sample_rate().get(), 48000);
    assert_eq!(rx.next(), Some(15.0));
    assert_eq!(rx.next(), Some(15.0));
    assert_eq!(rx.next(), Some(-5.0));
    assert_eq!(rx.next(), Some(-5.0));
    assert_eq!(rx.next(), Some(15.0));
    assert_eq!(rx.next(), Some(15.0));
    assert_eq!(rx.next(), Some(-5.0));
    assert_eq!(rx.next(), Some(-5.0));
    assert_eq!(rx.next(), None);
}

#[test]
fn rate_conv() {
    let (tx, mut rx) = mixer::mixer(nz!(1), nz!(96000));

    tx.add(SamplesBuffer::new(
        nz!(1),
        nz!(48000),
        vec![10.0, -10.0, 10.0, -10.0],
    ));
    tx.add(SamplesBuffer::new(
        nz!(1),
        nz!(48000),
        vec![5.0, 5.0, 5.0, 5.0],
    ));

    assert_eq!(rx.channels(), nz!(1));
    assert_eq!(rx.sample_rate().get(), 96000);
    assert_eq!(rx.next(), Some(15.0));
    assert_eq!(rx.next(), Some(5.0));
    assert_eq!(rx.next(), Some(-5.0));
    assert_eq!(rx.next(), Some(5.0));
    assert_eq!(rx.next(), Some(15.0));
    assert_eq!(rx.next(), Some(5.0));
    assert_eq!(rx.next(), Some(-5.0));
    assert_eq!(rx.next(), Some(-2.5));
    assert_eq!(rx.next(), None);
}

#[test]
fn start_afterwards() {
    let (tx, mut rx) = mixer::mixer(nz!(1), nz!(48000));

    tx.add(SamplesBuffer::new(
        nz!(1),
        nz!(48000),
        vec![10.0, -10.0, 10.0, -10.0],
    ));

    assert_eq!(rx.next(), Some(10.0));
    assert_eq!(rx.next(), Some(-10.0));

    tx.add(SamplesBuffer::new(
        nz!(1),
        nz!(48000),
        vec![5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 7.0],
    ));

    assert_eq!(rx.next(), Some(15.0)); // 10 + 5
    assert_eq!(rx.next(), Some(-5.0));

    assert_eq!(rx.next(), Some(6.0));
    assert_eq!(rx.next(), Some(6.0));

    tx.add(SamplesBuffer::new(nz!(1), nz!(48000), vec![2.0]));

    assert_eq!(rx.next(), Some(9.0));
    assert_eq!(rx.next(), Some(7.0));
    assert_eq!(rx.next(), Some(7.0));

    assert_eq!(rx.next(), None);
}

#[test]
fn added_taking_phase_into_account() {
    let (tx, mut rx) = mixer::mixer(nz!(2), nz!(48000));

    tx.add(SamplesBuffer::new(
        nz!(2),
        nz!(48000),
        vec![10.0, -10.0, 10.0, -10.0],
    ));

    assert_eq!(rx.next(), Some(10.0));

    tx.add(SamplesBuffer::new(
        nz!(2),
        nz!(48000),
        vec![5.0, -5.0, 6.0, -6.0],
    ));

    assert_eq!(rx.next(), Some(-10.0)); // not yet mixed (out of phase)
    assert_eq!(rx.next(), Some(15.0)); // mixing starts
}

