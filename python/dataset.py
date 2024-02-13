from path import Path

import numpy  as np
import polars as pl

from torch.utils.data import Dataset


class DS(Dataset):
    def __init__(self, path : Path, max_files=None):
        self.filenames          = sorted(path.glob("*.parquet"))
        if max_files is not None:
            self.filenames      = self.filenames[:max_files]
        self._n_per_file        = None
        self.pos                = None
        self.response           = None
        self.current_file_index = None

    def load_file(self, file_index):
        self.current_file_index = file_index
        df = pl.read_parquet(self.filenames[file_index])
        self.pos      = df.select(list("xy"))
        self.response = df.select("^sipm.*$")

    @property
    def n_per_file(self):
        if self._n_per_file is None:
            self.load_file(0)
            self._n_per_file = len(self.pos)

        return self._n_per_file

    def indices(self, f_train, f_valid):
        n_train = int(f_train / 100 * len(self))
        n_valid = int(f_valid / 100 * len(self))
        n_test  = len(self) - n_train - n_valid

        idx_train = np.arange(n_train)
        idx_valid = np.arange(n_valid) + n_train
        idx_test  = np.arange(n_test ) + n_train + n_valid
        return idx_train, idx_valid, idx_test

    def __len__(self):
        return self.n_per_file * len(self.filenames)

    def __getitem__(self, i):
        ifile = i // self.n_per_file
        ievt  = i  % self.n_per_file
        if ifile != self.current_file_index:
            self.load_file(ifile)

        pos      = np.array(list(self.pos     .row(ievt)))
        response = np.array(list(self.response.row(ievt)))
        return (pos, response)
