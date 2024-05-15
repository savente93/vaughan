#[macro_export]
macro_rules! assert_eq_fl {
    ($left:expr, $right:expr) => {
        let d = ($left as f64 - $right as f64).abs();
        assert!(
            d < 0.00001,
            "Expressions {} and {} differ by {}",
            $left,
            $right,
            d
        );
    };
    ($left:expr, $right:expr, $tol:expr) => {
        let d = ($left as f64 - $right as f64).abs();
        assert!(
            d < $tol,
            "Expressions {} and {} differ by {}",
            $left,
            $right,
            d
        );
    };
}

#[macro_export]
macro_rules! assert_sr_close {
    ($left:expr, $right:expr) => {
        assert_eq!($left.len(), $right.len());
        let diffs = ($left - $right).abs()?;
        let uneq = diffs.gt(0.0001)?;
        assert!(
            &uneq.filter(&uneq)?.len() == &0,
            "Expressions {:?} and {:?} differ by {:?} ",
            $left.filter(&uneq),
            $right.filter(&uneq),
            diffs.filter(&uneq)
        );
    };
    ($left:expr, $right:expr, $tol:expr) => {
        assert_eq!($left.len(), $right.len());
        let diffs = ($left - $right).abs()?;
        let uneq = diffs.gt($tol)?;
        assert!(
            &uneq.filter(&uneq)?.len() == &0,
            "Expressions {:?} and {:?} differ by {:?} ",
            $left.filter(&uneq),
            $right.filter(&uneq),
            diffs.filter(&uneq)
        );
    };
}
