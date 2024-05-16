use polars::frame::DataFrame;
use polars::prelude::*;

#[macro_export]
macro_rules! assert_eq_fl {
    ($left:expr, $right:expr) => {
        let ls = stringify!($left);
        let l = $left;
        let rs = stringify!($right);
        let r = $right;

        let d = (l as f64 - r as f64).abs();
        assert!(
            d < 0.00001,
            "Expressions {}={} and {}={} differ by {}",
            ls,
            l,
            rs,
            r,
            d
        );
    };
    ($left:expr, $right:expr, $tol:expr) => {
        let d = ($left as f64 - $right as f64).abs();
        let ls = stringify!($left);
        let l = $left;
        let rs = stringify!($right);
        let r = $right;
        assert!(
            d < $tol,
            "Expressions {}={} and {}={} differ by {}",
            ls,
            l,
            rs,
            r,
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

// courtesy of scikit-learn testing suite
// made with an svm, we don't do predictions here,
//so just include the result as it's fairly small
pub fn iris_predictions() -> DataFrame {
    df!(
    "prediction" => [ 2,2,1,2,2,2,0,1,2,2,2,2,0,2,2,0,2,2,2,2,1,2,2,2,0,2,0,2,1,1,2,0,2,1,0,0,0,0,2,0,2,1,0,2,0,0,0,0,1,2,2,2,0,2,2,2,0,0,2,1,2,0,2,2,0,2,2,2,2,0,2,0,2,2,2 ],
    "truth" => [ 2,2,0,2,1,1,0,1,2,1,2,1,1,1,1,0,2,2,1,0,2,1,2,2,0,1,0,2,1,0,1,0,1,1,0,0,0,0,2,0,1,2,0,1,0,1,1,0,0,1,1,1,1,2,1,1,0,0,2,0,1,0,2,2,0,1,1,1,1,0,2,0,1,2,2 ],
    ).unwrap()
}

pub fn iris_predictions_binary() -> DataFrame {
    df!(
    "prediction" => [ 0,0,1,1,0,0,1,1,0,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,0 ],
    "truth" =>      [ 0,0,1,1,1,0,1,0,1,0,1,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,0,1,0,0,1,1,1,0 ],
    ).unwrap()
}
