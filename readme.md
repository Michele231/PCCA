![alt text](https://github.com/Michele231/PCCA/blob/main/figures/suc.PNG)

***

# Principal Compontents Classification Algorithm (Work in Progress!)

This is a Work in Progress project! The "distributional method" as defined in Maestri et al., (2019) is not yet implemented! The algorithm works, but the performance could be bad (mostly when the two training sets have different dimension).

### Installation

```bash
git clone https://github.com/Michele231/PCCA.git
cd PCCA
make install
```

### Dependencies

1. Eigen (tested on v3.4.0)
2. Spectra (tested on v1.0.0)

### Usage

#### Train

Put the two training sets in the "input" folder with the names "TS1.txt" and "TS2.txt". The training sets have 

```bash
./cic train
```

#### Test and Run

Put the data to analize in the "input" folder with the name "data.txt" together with the training sets "TS1.txt" and "TS2.txt". 

```bash
./cic test
```

#### Results
The classification of the data is in the output folder. A positive number indicates a similarity with the class number one (TS1). On the other hand, a negative number indicates a similarity with the class number two (TS2).

***

![alt text](https://github.com/Michele231/cic/blob/main/figures/suc.PNG)


