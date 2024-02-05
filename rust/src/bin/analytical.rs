use std::io;
use std::io::{Write, BufWriter};
use std::fs::File;
use std::path::PathBuf;


use clap::Parser;
use rand::random;
use indicatif::ProgressBar;
use itertools::Itertools;

/// CLI
#[derive(clap::Parser, Debug, Clone)]
#[clap( name  = "analytical"
      , about = "Create data based on an analytical")]
pub struct Cli {
    #[clap(short = 'o', long, default_value = "out.csv")]
    pub outfile : PathBuf,

    #[clap(short = 'j', long, default_value = "4")]
    pub threads : u8,

    #[clap(short = 'b', long, default_value = "10")]
    pub base : u8,

    #[clap(short = 'e', long)]
    pub exponent : u8,
}

struct Event {
    x: f32,
    y: f32,
    z: f32,
    response : Vec<f32>,
}

struct Pos(f32, f32, f32);

fn generate_random_position() -> Pos {
    let r  = 200f32;
    let r2 = r*r;
    loop {
        let x = r * (random::<f32>() - 0.5);
        let y = r * (random::<f32>() - 0.5);
        if x*x + y*y < r2 {
            let z = 500f32 * random::<f32>();
            return Pos(x, y, z);
        }
    }
}

fn psf(x1 : f32, y1 : f32, x0 : f32, y0 : f32) -> f32 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dz = 5f32;
    dz.powf(1.5) / (dx*dx + dy*dy + dz*dz).powf(1.5)
}

fn apply_psf(pos : Pos, sipm_pos : &Vec<(f32, f32)>) -> Event {
    let Pos(x, y, z) = pos;
    let response : Vec<f32> =
    sipm_pos.iter()
            .map(|(xs, ys)| { psf(*xs, *ys, x, y) })
            .collect();

    Event{x, y, z, response}
}

struct CsvWriter {
    writer   : BufWriter<File>,
}

impl CsvWriter {
    pub fn new(filename : PathBuf) -> Self {
        let file   = File::create(filename).unwrap();
        let writer = BufWriter::new(file);
        Self{writer}
    }

    pub fn write(&mut self, evt : &Event) -> io::Result<()> {
        let response = evt.response.iter()
            .map(f32::to_string)
            .collect::<Vec<String>>()
            .join(", ");
        let line = format!("{} {} {} {}\n", evt.x, evt.y, evt.z, response);
        self.writer.write(&line.into_bytes())?;
        Ok(())
    }
}

fn sipm_positions() -> Vec<(f32, f32)> {
    (0..45).cartesian_product(0..45)
           .map(|(i, j)| {(-220 + i*10, -200 + j*10)})
           .map(|(x, y)| {(   x as f32,    y as f32)})
           .collect()
}


fn main() -> io::Result<()> {
    let args = Cli::parse();
    let filename = PathBuf::from(&args.outfile);
    std::fs::create_dir_all(filename.parent().unwrap())
        .unwrap_or_else(|e| panic!("Cannot write to destination {}: {e}", args.outfile.display()));

    let mut writer = CsvWriter::new(filename);
    let sipm_pos   = sipm_positions();

    let n    = (args.base as u32).pow(args.exponent as u32) as u64;
    let pmod = n / 200;
    let pb   = ProgressBar::new(n);
    (0..n).into_iter()
          .map     (|i  | { if i % pmod == 0 {pb.inc(pmod);} })
          .map     (|_  | { generate_random_position()           })
          .map     (|pos| { apply_psf(pos, &sipm_pos)       })
          .for_each(|evt| { writer.write(&evt).expect("") });

    Ok(())
}
