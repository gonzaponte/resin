use std::fs::File;
use std::path::PathBuf;

use clap::Parser;
use polars::prelude::*;
use indicatif::ProgressBar;
use itertools::Itertools;
use rand::random;

/// CLI
#[derive(clap::Parser, Debug, Clone)]
#[clap( name  = "analytical"
      , about = "Create data based on an analytical")]
pub struct Cli {
    #[clap(short = 'o', long, default_value = "out.parquet")]
    pub outfile : PathBuf,

    #[clap(short = 'f', long, default_value = "1000")]
    pub evt_per_file : u64,

    #[clap(short = 'j', long, default_value = "4")]
    pub threads : u8,

    #[clap(short = 'b', long, default_value = "10")]
    pub base : u64,

    #[clap(short = 'e', long)]
    pub exponent : u32,

    #[clap(long, default_value = "10000")]
    pub rmax : f64,
}

fn new_filename(filename : PathBuf, index : u64) -> PathBuf {
    let filename    = filename.into_os_string().into_string().unwrap();
    let split_index = filename.rfind(".").expect("Pattern not found");
    let (basename, extension) =  filename.split_at(split_index);
    let new_file = PathBuf::from(format!("{basename}_{index}{extension}"));
    new_file
}

fn generate_random_position() -> (f64, f64, f64) {
    let r  = 200f64;
    let r2 = r*r;
    loop {
        let x = r * (random::<f64>() - 0.5);
        let y = r * (random::<f64>() - 0.5);
        if x*x + y*y < r2 {
            let z = 500f64 * random::<f64>();
            return (x, y, z);
        }
    }
}

fn psf(x1 : f64, y1 : f64, x0 : f64, y0 : f64, rmax2 : f64) -> f64 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dz = 5f64;
    let dr2 = dx*dx + dy*dy;
    if dr2 > rmax2 {0.} else {
        dz.powf(1.5) / (dr2 + dz*dz).powf(1.5)
    }
}

fn apply_psf(pos : (f64, f64, f64), sipm_pos : &Vec<(f64, f64)>, rmax2 : f64) -> (f64, f64, f64, Vec<f64>) {
    let (x, y, z) = pos;
    let response : Vec<f64> =
    sipm_pos.iter()
            .map(|(xs, ys)| { psf(*xs, *ys, x, y, rmax2) })
            .collect();

    (x, y, z, response)
}

fn create_df(data : (f64, f64, f64, Vec<f64>)) -> LazyFrame {
    let response = data.3
        .into_iter()
        .enumerate()
        .map(|(i,v)| { (format!("sipm_{}", i).to_string(),  v) } )
        .map(|(name, value)| { Series::new(&name, &[value]) } )
        ;

    let mut columns = vec![
        Series::new("x", &[data.0]),
        Series::new("y", &[data.1]),
        Series::new("z", &[data.2]),
    ];
    columns.extend(response);
    DataFrame::new(columns).unwrap().lazy()
}

fn sipm_positions() -> Vec<(f64, f64)> {
    (0..45).cartesian_product(0..45)
           .map(|(i, j)| {(-220 + i*10, -200 + j*10)})
           .map(|(x, y)| {(   x as f64,    y as f64)})
           .collect()
}



fn main() -> Result<(), String> {
    let args = Cli::parse();
    println!("{:?}", args);

    let filename = PathBuf::from(&args.outfile);
    std::fs::create_dir_all(filename.parent().expect("Could not access parent directory"))
        .expect("Cannot write to destination");

    new_filename(filename.clone(), 123);

    let sipm_pos = sipm_positions();

    let ntot   = args.base.pow(args.exponent);
    let nfile  = args.evt_per_file.min(ntot);
    let nbatch = (ntot as f64 / nfile as f64).round() as u64;
    assert_eq!(nbatch * nfile, ntot, "Invalid ratio of evt_per_file and ntot");

    let rmax2 = args.rmax * args.rmax;
    let pb    = ProgressBar::new(nbatch);
    (0..nbatch).into_iter()
        .inspect(|_| { pb.inc(1);} )
        .map    (|i| new_filename(args.outfile.clone(), i))
        .for_each(|filename| {
            let dfs : Vec<LazyFrame> =
            (0..nfile).into_iter()
                       .map     (|_  | { generate_random_position()       })
                       .map     (|pos| { apply_psf(pos, &sipm_pos, rmax2) })
                       .map     (create_df)
                       .collect();

            let mut df = concat(&dfs, UnionArgs::default()).unwrap().collect().unwrap();

            let mut file = File::create(filename).unwrap();
            ParquetWriter::new(&mut file).finish(&mut df).unwrap();
        });
    Ok(())

}
