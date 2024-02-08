use std::fs::File;
use std::path::{Path, PathBuf};

use clap::Parser;
use polars::prelude::*;
use indicatif::ProgressBar;
use itertools::Itertools;
use rand::random;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

/// CLI
#[derive(clap::Parser, Debug, Clone)]
#[clap( name  = "analytical"
      , about = "Create data based on an analytical")]
pub struct Cli {
    /// Out put file
    #[clap(short = 'o', long, default_value = "data/out.parquet")]
    pub outfile: PathBuf,

    /// Number of events per file
    #[clap(short = 'f', long, default_value = "1000")]
    pub evt_per_file: u64,

    /// Number of threads to use
    #[clap(short = 'j', long, default_value = "4")]
    pub threads: u8,

    /// Base of the total number of events in n=base**exponent
    #[clap(short = 'b', long, default_value = "10")]
    pub base: u64,

    /// Exponent of the total number of events in n=base**exponent
    #[clap(short = 'e', long)]
    pub exponent: u32,

    /// Exponent of the total number of events in n=base**exponent
    #[clap(short = 's', long, default_value = "0")]
    pub offset: u64,

    /// Active volume radius in mm
    #[clap(short='r', long, default_value = "200")]
    pub r: f64,

    /// Maximum radius for PSF application in mm
    #[clap(long, default_value = "10000")]
    pub drmax: f64,

    /// Flag: Print SiPM positions.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    pub print_sipm_positions: bool,
}

fn new_filename(filename : &Path, index : u64) -> PathBuf {
    let filename    = filename.to_str().unwrap();
    let split_index = filename.rfind(".").expect("Pattern not found");
    let (basename, extension) = filename.split_at(split_index);
    let new_file = PathBuf::from(format!("{basename}_{index}{extension}"));
    new_file
}

fn generate_random_position(r: f64) -> (f64, f64, f64) {
    let r2 = r*r;
    loop {
        let x = 2. * r * (random::<f64>() - 0.5);
        let y = 2. * r * (random::<f64>() - 0.5);
        if x*x + y*y < r2 {
            let z = 500f64 * random::<f64>();
            return (x, y, z);
        }
    }
}

fn psf(x1: f64, y1: f64, x0: f64, y0: f64, rmax2: f64, drmax2: f64) -> f64 {
    if  x0*x0 + y0*y0 > rmax2 { return 0_f64; }
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dz = 5f64;
    let dr2 = dx*dx + dy*dy;
    if dr2 > drmax2 { return 0.}

    dz.powf(1.5) / (dr2 + dz*dz).powf(1.5)
}

fn apply_psf(pos: (f64, f64, f64), sipm_pos: &Vec<(f64, f64)>, rmax2: f64, drmax2: f64) -> (f64, f64, f64, Vec<f64>) {
    let (x, y, z) = pos;
    let response: Vec<f64> =
        sipm_pos.iter()
            .map(|(xs, ys)| { psf(*xs, *ys, x, y, rmax2, drmax2) })
            .collect();

    (x, y, z, response)
}

fn create_df(data: (f64, f64, f64, Vec<f64>)) -> LazyFrame {
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

fn sipm_positions(n: usize, p: f64) -> Vec<(f64, f64)> {
    let nf = n as f64;
    let x0 = -( p/2. + (nf-2.)/2.*p);
    (0..n).cartesian_product(0..n)
           .map(|(i, j)| {( i as f64, j as f64)})
           .map(|(i, j)| {( x0 + i*p, x0 + j*p)})
           .collect()
}



fn main() -> Result<(), String> {
    let args = Cli::parse();
    println!("{:?}", args);

    let filename = args.outfile;
    std::fs::create_dir_all(&filename.parent().expect("Could not access parent directory"))
        .expect("Cannot write to destination");

    let n = 64;
    let p = 10.;
    let sipm_pos = sipm_positions(n, p);
    if args.print_sipm_positions {
        for (x, y) in &sipm_pos {
            println!("{x} {y}")
        }
    }

    let ntot   = args.base.pow(args.exponent);
    let nfile  = args.evt_per_file.min(ntot);
    let nbatch = (ntot as f64 / nfile as f64).round() as u64;
    assert_eq!(nbatch * nfile, ntot, "Invalid ratio of evt_per_file and ntot");

    let r      = args.r;
    let  rmax2 = r * r;
    let drmax2 = args.drmax * args.drmax;
    let pb     = ProgressBar::new(nbatch); pb.set_position(0);
    ThreadPoolBuilder::new()
        .num_threads(args.threads as usize)
        .build_global()
        .unwrap();

    (0..nbatch).into_par_iter()
        .map(|i| new_filename(&filename, i + args.offset))
        .map(|filename| {
            let dfs: Vec<LazyFrame> =
            (0..nfile).into_iter()
                      .map     (|_  | { generate_random_position(r)              })
                      .map     (|pos| { apply_psf(pos, &sipm_pos, rmax2, drmax2) })
                      .map     (create_df)
                      .collect();

            let mut df = concat(&dfs, UnionArgs::default()).unwrap().collect().unwrap();

            let mut file = File::create(filename).unwrap();
            ParquetWriter::new(&mut file).finish(&mut df).unwrap();
        })
        .for_each(|_| { pb.inc(1);});

    Ok(())
}
