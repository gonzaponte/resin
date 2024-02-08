use resin::point::Point;

use clap::Parser;
use polars::prelude::*;
use indicatif::ProgressBar;
use itertools::Itertools;
use rand::random;
//use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use std::fs::File;
use std::path::{Path, PathBuf};

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
    pub file_offset: u64,

    /// Active volume radius in mm
    #[clap(short='r', long, default_value = "200")]
    pub r: f64,

    /// Flag: Print SiPM positions.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    pub print_sipm_positions: bool,
}

fn new_filename(filename: &Path, index: u64) -> PathBuf {
    let filename    = filename.to_str().unwrap();
    let split_index = filename.rfind(".").expect("Pattern not found");
    let (basename, extension) = filename.split_at(split_index);
    let new_file = PathBuf::from(format!("{basename}_{index}{extension}"));
    new_file
}

fn generate_random_position(half_range: f64) -> Point {
    let x = half_range * (random::<f64>() - 0.5);
    let y = half_range * (random::<f64>() - 0.5);
    Point{x, y}
}

fn psf(p1: Point, p0: Point) -> f64 {
    let dp = p1 - p0;
    let dz = 5_f64;

    dz.powf(1.5) / (dp.r2() + dz*dz).powf(1.5)
}


fn apply_psf(p0: Point, refp: Point, sipm_pos: &Vec<Point>, rmax2: f64) -> Vec<f64> {
    let mut response: Vec<f64> =
        sipm_pos.iter()
                .map(|&ps| {
                    if   (ps + refp).r2() < rmax2 { psf(ps, p0) }
                    else                          {      0_f64  }
                })
                .collect();

    response.insert(0, p0.y + refp.y);
    response.insert(0, p0.x + refp.x);
    response.insert(0, p0.y);
    response.insert(0, p0.x);
    response
}

fn create_df(data: Vec<Vec<f64>>, names: &Vec<String>) -> DataFrame {
    let get_field = |i| {
        data.iter()
            .map(|v| v[i])
            .collect::<Vec<f64>>()
    };
    let columns: Vec<Series> =
    names.iter()
         .enumerate()
         .map(|(i, name)| (name, get_field(i)))
         .map(|(name, v)| Series::new(&name, &v))
         .collect();

    DataFrame::new(columns).unwrap()
}

fn sipm_positions(n: usize, p: f64) -> Vec<Point> {
    let nf = n as f64;
    let x0 = -( p/2. + (nf-2.)/2.*p);
    (0..n).cartesian_product(0..n)
           .map(|(i, j)| {( i as f64, j as f64)})
           .map(|(i, j)| {( x0 + i*p, x0 + j*p)})
           .map(|(x, y)| {Point{x, y}})
           .collect()
}

fn pick_ref(p: f64, rmax: f64) -> Point {
    loop {
        let Point{mut x, mut y} = generate_random_position(rmax + p);
        x = x.div_euclid(p);
        y = y.div_euclid(p);
        let p = Point{x, y};
        if p.r() < rmax {return p;}
    }
}


fn main() -> Result<(), String> {
    let args = Cli::parse();
    println!("{:?}", args);

    let filename = args.outfile;
    std::fs::create_dir_all(&filename.parent().expect("Could not access parent directory"))
        .expect("Cannot write to destination");

    let nsipms   = 16;
    let pitch    = 10.;
    let sipm_pos = sipm_positions(nsipms, pitch);
    if args.print_sipm_positions {
        for pos in &sipm_pos {
            println!("{} {}", pos.x, pos.y)
        }
    }

    let ntot   = args.base.pow(args.exponent);
    let nfile  = args.evt_per_file.min(ntot);
    let nbatch = (ntot as f64 / nfile as f64).round() as u64;
    assert_eq!(nbatch * nfile, ntot, "Invalid ratio of evt_per_file and ntot");

    let r  = args.r;
    let r2 = r * r;
    let pb = ProgressBar::new(nbatch); pb.set_position(0);
    ThreadPoolBuilder::new()
        .num_threads(args.threads as usize)
        .build_global()
        .unwrap();

    let mut column_names = Vec::with_capacity(4 + nsipms);
    column_names.push("x".to_string());
    column_names.push("y".to_string());
    column_names.push("xabs".to_string());
    column_names.push("yabs".to_string());
    (0..nsipms).into_iter()
               .map(|i| format!("sipm_{i}").to_string())
               .for_each(|name| column_names.push(name));

    (0..nbatch).into_iter()
        .map(|i| new_filename(&filename, i + args.file_offset))
        .map(|filename| {
            let data: Vec<Vec<f64>> =
            (0..nfile).into_iter()
                      .map     (|_          | { (generate_random_position(pitch/2.), pick_ref(pitch, r)) })
                      .inspect (|(pos, refp)| { println!("{:?} {:?}", pos, refp)                         })
                      .map     (|(pos, refp)| { apply_psf(pos, refp, &sipm_pos, r2)                      })
                      .collect();

            let mut df   = create_df(data, &column_names);
            let mut file = File::create(filename).unwrap();
            ParquetWriter::new(&mut file).finish(&mut df).unwrap();
        })
        .for_each(|_| { pb.inc(1);});

    Ok(())
}
