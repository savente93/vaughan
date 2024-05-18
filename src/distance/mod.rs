use polars::prelude::*;

use crate::utils::extract_numeric;

use polars::error::PolarsResult as Result;
use polars::prelude::LazyFrame;

pub fn euclidian(data: LazyFrame, left: &str, right: &str) -> Result<f64> {
    let d = data
        .select(&[(col(left) - col(right)).alias("_diff")])
        .select(&[(col("_diff") * col("_diff")).alias("_diff_sq")])
        .select(&[col("_diff_sq").sum()])
        .collect()?;
    Ok(extract_numeric(&d.get(0).unwrap()[0])?.sqrt())
}

pub fn manhattan(data: LazyFrame, left: &str, right: &str) -> Result<f64> {
    let d = data
        .select(&[(col(left) - col(right)).abs().sum()])
        .collect()?;
    extract_numeric(&d.get(0).unwrap()[0])
}

pub fn chebyshev(data: LazyFrame, left: &str, right: &str) -> Result<f64> {
    let d = data
        .select(&[(col(left) - col(right)).abs().max()])
        .collect()?;
    extract_numeric(&d.get(0).unwrap()[0])
}
pub fn hamming(data: LazyFrame, left: &str, right: &str) -> Result<f64> {
    let d = data.select(&[col(left).neq(col(right)).sum()]).collect()?;
    extract_numeric(&d.get(0).unwrap()[0])
}

#[cfg(test)]
mod test {
    use crate::assert_eq_fl;

    use super::*;
    #[test]
    fn trivial_euclidean() -> Result<()> {
        let df = df!("x" => [1,0], "y"=> [0,1])?;
        assert_eq_fl!(euclidian(df.lazy(), "x", "y")?, 2.0_f64.sqrt());
        Ok(())
    }
    #[test]
    fn trivial_chebyshev() -> Result<()> {
        let df = df!("x" => [1,0], "y"=> [0,1])?;
        assert_eq_fl!(chebyshev(df.lazy(), "x", "y")?, 1.0_f64);
        Ok(())
    }
    #[test]
    fn trivial_manhattan() -> Result<()> {
        let df = df!("x" => [1,0], "y"=> [0,1])?;
        assert_eq_fl!(manhattan(df.lazy(), "x", "y")?, 2.0_f64);
        Ok(())
    }
    #[test]
    fn trivial_hamming() -> Result<()> {
        let df = df!("x" => [1,0], "y"=> [0,1])?;
        assert_eq_fl!(hamming(df.lazy(), "x", "y")?, 2.0_f64);
        Ok(())
    }
}
