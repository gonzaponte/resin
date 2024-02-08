use std::ops::{Add, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Point{
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn r (&self) -> f64 { self.r2().powf(0.5) }
    pub fn r2(&self) -> f64 { self.x * self.x + self.y * self.y }
}

impl Add<Point> for Point {
    type Output = Self;

    fn add(self, rhs: Point) -> Self::Output {
        Point{ x: self.x + rhs.x
             , y: self.y + rhs.y}
    }
}

impl Add<f64> for Point {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        Point{ x: self.x + rhs
             , y: self.y + rhs}
    }
}

impl Sub<Point> for Point {
    type Output = Self;

    fn sub(self, rhs: Point) -> Self::Output {
        Point{ x: self.x - rhs.x
             , y: self.y - rhs.y}
    }
}

impl Sub<f64> for Point {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self::Output {
        Point{ x: self.x - rhs
             , y: self.y - rhs}
    }
}
