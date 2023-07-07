/* This code is a C++ version of the matlab CIC developed by Maestri et al 2019 */
/* Author:  Michele Martinazzo                                                  */
/* OG Code: Tiziano Maestri                                                     */
/* Version: 0.02 (June 2023)                                                    */

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <vector>
#include <float.h>
#include <Eigen/Dense>
#include <Spectra/SymEigsSolver.h>

// method to read and write binary files  

namespace Eigen{
template<class Matrix>
void write_binary(const char* filename, const Matrix& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}
template<class Matrix>
void read_binary(const char* filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
}
} // 

using namespace std;
using namespace Eigen;


//##########################################################################
struct EingenValVec {
    MatrixXd covMatrx;
    VectorXd eigenVal;
    MatrixXd eigenVec;
};
//##########################################################################
class CIC {
    public:
    CIC(){};
    void train(const char *f1, const char *f2);
    void test(const char *f1, const char *f2, const char *test_set_name);

    private:
    MatrixXd readMatrix(const char *filename);
    EingenValVec computeEigen(const Eigen::MatrixXd& inputMatrix);
    EingenValVec computeSpecrtralEigen(const Eigen::MatrixXd& inputMatrix,
                                    const unsigned int np0);
    int significant_reshape(Eigen::VectorXd& eingVal,
        Eigen::MatrixXd& eingVec, const int n_realizations);
    MatrixXd removeRow(const MatrixXd& OGmatrix, unsigned int rowToRemove);
    MatrixXd addRow(const MatrixXd& OGmatrix, const MatrixXd& rowToAdd);
    vector<double> distMethod(const MatrixXd& TSa, const MatrixXd& TSb, 
                    const EingenValVec& eigTSa, const EingenValVec& eigTSb, 
                    unsigned int p0);
    double best_HSS(vector<double>& SIDa, vector<double>& SIDb);
    vector<double> testValues(const MatrixXd& TSa, const MatrixXd& TSa_eigenVec,
                          const MatrixXd& TestS, unsigned int p0);
};
//##########################################################################
// Methods of the class

// method read matrix
MatrixXd CIC::readMatrix(const char *filename){
    // It works with files containing matrices of arbitrary size.
    int cols = 0, rows = 0;
    const unsigned int BIGBUFFSIZE = 500000; // to be subtituted to BUFSIZ
    double buff[BIGBUFFSIZE]; //BUFSIZ represents the recommended buffer size for input/outpu

    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (! infile.eof()){
        string line;
        getline(infile, line);

        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];
        if (temp_cols == 0)
            continue;
        if (cols == 0)
            cols = temp_cols;

        rows++;
        }
    infile.close();
    rows--;

    // Populate matrix with numbers.
    MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];

    return result;
    }

//##########################################################################
//method compute the eingenVecVal
EingenValVec CIC::computeEigen(const Eigen::MatrixXd& inputMatrix) {
    // This function compute the eigenvalues and vectors of the cov matrix
    // of the input nxm matrix, where n are the different realization of
    // the observation and m are the dimension of the measure

    // Compute the mean of the input matrix
    VectorXd mean = inputMatrix.colwise().mean();
    // Subtract the mean from each data point
    MatrixXd centered = inputMatrix.rowwise() - mean.transpose();
    // Compute the covariance matrix
    MatrixXd covariance = (centered.transpose()*centered)/double(inputMatrix.rows()-1);

    //

    // Solver
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covariance);

    if (eigensolver.info() != Eigen::Success) {
        cerr << "Eigenvalue decomposition failed." << endl;
       EingenValVec Vuoto;
        return Vuoto;
    };

    
    EingenValVec results;

    results.covMatrx = covariance;
    results.eigenVal = eigensolver.eigenvalues();
    results.eigenVec = eigensolver.eigenvectors();

    return results;
}

//##########################################################################
//method compute the eingenVecVal
EingenValVec CIC::computeSpecrtralEigen(const Eigen::MatrixXd& inputMatrix,
                                        const unsigned int np0) {

    // Compute the mean of the input matrix
    VectorXd mean = inputMatrix.colwise().mean();
    // Subtract the mean from each data point
    MatrixXd centered = inputMatrix.rowwise() - mean.transpose();
    // Compute the covariance matrix
    MatrixXd covariance = (centered.transpose()*centered)/double(inputMatrix.rows()-1);

    // Construct matrix operation object using the wrapper class DenseSymMatProd
    Spectra::DenseSymMatProd<double> op(covariance);
    // Construct eigen solver object, requesting the largest np0 eigenvalues
    Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, np0, 2*np0+1);

    // Initialize and compute
    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::LargestAlge);
 
    // Retrieve results
    EingenValVec results;

    if (eigs.info() == Spectra::CompInfo::Successful) {
        results.covMatrx = covariance;
        results.eigenVal = eigs.eigenvalues();
        results.eigenVec = eigs.eigenvectors();
    };

    return results;

}

//##########################################################################
//method cut the eigen values
int CIC::significant_reshape(Eigen::VectorXd& eingVal,
        Eigen::MatrixXd& eingVec, const int n_realizations){
    
    int rankm = eingVal.rows();
    if (n_realizations<eingVal.rows()) rankm=n_realizations; // max rank of data matrix
    
    // cut the eingVal and vect
    //int n_signific = 0;
    //while (eingVal[n_signific]<signeig) {
    //    n_signific += 1;
    //};
    //cout << "n_signific: " << n_signific << endl;
    
    //reshaping the eigenvalues
    VectorXd eingVal_temp = eingVal.reverse();
    eingVal_temp.conservativeResize(rankm);
    eingVal = eingVal_temp;

    //reshaping the eigenvectors
    MatrixXd eingVec_temp = eingVec.rowwise().reverse();
    eingVec_temp.conservativeResize(eingVec_temp.rows(),
                                    rankm);
    eingVec = eingVec_temp;

    // Computing the number of information bearing components (IBECs)
    // by Turner et al. (2006) 
    double re, ind, ind2;
    int p0 = 0;
    ind2 = DBL_MAX;
    for (int i=1;i<rankm;i++) {
        re = 0.0;
        for (int j=0;j<rankm-i;j++) {
            re += eingVal.reverse()[j];
        }
        re = sqrt(re/(double(n_realizations)*(double(rankm)-double(i))));
        ind = re/pow(double(rankm)-double(i),2.);

        if (ind2>ind) {
            ind2 = ind; // select the lower value
            p0 = i;
        }
    }

    return p0;
}

//##########################################################################
// Method to remove a row
MatrixXd CIC::removeRow(const MatrixXd& OGmatrix, unsigned int rowToRemove){
    MatrixXd matrix = OGmatrix;
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = 
        matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
    return matrix;
    };

//##########################################################################
// Method to add a row
MatrixXd CIC::addRow(const MatrixXd& OGmatrix,const MatrixXd& rowToAdd){
    MatrixXd matrix = OGmatrix;
    unsigned int numRows = matrix.rows()+1;
    unsigned int numCols = matrix.cols();

    matrix.conservativeResize(numRows,numCols);
    matrix.row(numRows-1) = rowToAdd;

    return matrix;
    }

//##########################################################################
// Distributional Method
vector<double> CIC::distMethod(const MatrixXd& TSa, const MatrixXd& TSb, 
                    const EingenValVec& eigTSa, const EingenValVec& eigTSb, 
                    unsigned int p0) {
    
    vector<double> SIa;
    vector<double> SIb;
    vector<double> SID;
    double sumETSmTS;
    double sumPP;

    for (int i = 0; i<TSb.rows(); i++) {

    MatrixXd RTSb = removeRow(TSb,i);
    MatrixXd ETSa = addRow(TSa,TSb.row(i));

    // Computing the Einjenvalues and vectors
    EingenValVec eig_RTSb = computeSpecrtralEigen(RTSb,p0);
    EingenValVec eig_ETSa = computeSpecrtralEigen(ETSa,p0);

    // Computing the similarity index from Cossich et al. (2021)
    sumPP = 0.0;
    for (int pp = 0; pp<p0; pp++) {
        sumETSmTS = 0.0;
        for (int nu = 0; nu<eig_ETSa.eigenVec.rows(); nu++) {
            sumETSmTS += abs(pow(eig_ETSa.eigenVec.col(pp)[nu],2.)-
            pow(eigTSa.eigenVec.col(pp)[nu],2.));
        };
        sumPP += (1./(2.*double(p0)))*sumETSmTS;
    };

    SIa.push_back(1.-sumPP); // vector conitaining the SI for the different combinations

    // Compute the similarity index (SI) for the reduced TS
    sumPP = 0.0;
    for (int pp = 0; pp<p0; pp++) {
        sumETSmTS = 0.0;
        for (int nu = 0; nu<eig_RTSb.eigenVec.rows(); nu++) {
            sumETSmTS += abs(pow(eig_RTSb.eigenVec.col(pp)[nu],2.)-
            pow(eigTSb.eigenVec.col(pp)[nu],2.));
        };
        sumPP += (1./(2.*double(p0)))*sumETSmTS;
    };

    SIb.push_back(1.-sumPP); // vector conitaining the SI for the different combinations

    // SID
    SID.push_back(SIb[i]-SIa[i]);

    };

    return SID;
}

//##########################################################################
// method to compute the translation to obtain the best HSS
double CIC::best_HSS(vector<double>& SIDa, vector<double>& SIDb) {

    for (int i = 0; i<SIDb.size(); i++){SIDb[i] = -SIDb[i];};

    double trlsl = -0.5; // initial tralslation
    int TP,TN,FP,FN;     // CM values    

    for (int i = 0; i == 1001; i++) {
        
        // definition of TP...
        // computation of HSS...
        
        trlsl += 0.001;
    };

    // find the higher HSS...

    return trlsl;
}


//##########################################################################
// method to analyze the test set
vector<double> CIC::testValues(const MatrixXd& TSa, const MatrixXd& TSa_eigenVec,
                          const MatrixXd& TestS, unsigned int p0){
    vector<double> SIa;
    double sumETSmTS;
    double sumPP;

    for (int i = 0; i<TestS.rows(); i++) {
    
    MatrixXd ETSa = addRow(TSa,TestS.row(i));

    // Computing the Einjenvalues and vectors
    EingenValVec eig_ETSa = computeSpecrtralEigen(ETSa, p0);

    // Compute the similarity index (SI) for the extended TS
    sumPP = 0.0;
    for (int pp = 0; pp<p0; pp++) {
        sumETSmTS = 0.0;
        for (int nu = 0; nu<eig_ETSa.eigenVec.rows(); nu++) {
            sumETSmTS += abs(pow(eig_ETSa.eigenVec.col(pp)[nu],2.)-
            pow(TSa_eigenVec.col(pp)[nu],2.));
        };
        sumPP += (1./(2.*double(p0)))*sumETSmTS;
    };

    SIa.push_back(1.-sumPP); // vector conitaining the SI for the different combinations


    };
    return SIa;

}

//##########################################################################
// Method to train the CIC
void CIC::train(const char *f1, const char *f2) {
    // reading the Training Sets
    MatrixXd TS1 = readMatrix(f1);
    MatrixXd TS2 = readMatrix(f2);

    // Useful info
    cout << endl;
    cout << "Training Set Info:" << endl;
    cout << TS1.cols() << " channels." <<endl;
    cout << TS1.rows() << " realizations." << endl;

    // Computing the Einjenvalues and vectors
    EingenValVec eig_T1 = computeEigen(TS1);
    EingenValVec eig_T2 = computeEigen(TS2);

    // Keep only the significant eigenvalues and vectors (eigval>signeig)
    //double const signeig = 1.e-5;
    int p01 = significant_reshape(eig_T1.eigenVal,eig_T1.eigenVec
                                        ,TS1.rows());
    int p02 = significant_reshape(eig_T2.eigenVal,eig_T2.eigenVec
                                        ,TS2.rows());

    int p0 = p01;
    if (p02<p01) {p0=p02;} // lower

    // Similarity of the class 1 to itself with respect to the class 2
    vector<double> SID1 = distMethod(TS2,TS1,eig_T2,eig_T1,p0);
    // Similarity of the class 2 to itself with respect to the class 1
    vector<double> SID2 = distMethod(TS1,TS2,eig_T1,eig_T2,p0);

    // Computing the translation (distributional method)
    // 
    double trlsl = best_HSS(SID1,SID2);


    // Save the data on a temp file to use them 
    write_binary("temp/TS1_eigenVec.dat",eig_T1.eigenVec);
    write_binary("temp/TS2_eigenVec.dat",eig_T2.eigenVec);
    // Save p0 on a temp txt file 
 	ofstream fout;
	fout.open("temp/TS12_p0.txt");   
    fout<<p0<<endl;
    fout.close();

    return;

}

//##########################################################################
void CIC::test(const char *f1, const char *f2, 
                const char *test_set_name) {
    // reading the Training Sets from the saved file
    MatrixXd TS1_eingVec_saved;
    MatrixXd TS2_eingVec_saved;
    read_binary("temp/TS1_eigenVec.dat",TS1_eingVec_saved);
    read_binary("temp/TS2_eigenVec.dat",TS2_eingVec_saved);
    // reading the p0 value
    ifstream fin;
    fin.open("temp/TS12_p0.txt");
    unsigned int p0;
    fin >> p0;
    fin.close();
    // reading the training and test sets
    MatrixXd TS1 = readMatrix(f1);
    MatrixXd TS2 = readMatrix(f2);
    MatrixXd test_set = readMatrix(test_set_name);

    // Useful info
    cout << endl;
    cout << "Test Set Info:" << endl;
    cout << test_set.cols() << " channels." <<endl;
    cout << test_set.rows() << " data." << endl;

    vector<double> SI1 = testValues(TS1, TS1_eingVec_saved,
                          test_set, p0);
    vector<double> SI2 = testValues(TS2, TS2_eingVec_saved,
                          test_set, p0);

    // Save the classification results
    ofstream fout;
    fout.open("output/results.txt");
    for (int i = 0;i<test_set.rows();i++) {
        fout << SI1[i]-SI2[i] << endl;
    }
    fout.close();

    return;
}
//##########################################################################

//##########################################################################
int main(int argc, char **argv) {

    string opt1 = "train";
    string opt2 = "test";

    CIC myCIC;
    if (!opt1.compare(argv[1])){
        cout << "Training the model" << endl;
        myCIC.train("input/TS1.txt","input/TS2.txt");
    } else if (!opt2.compare(argv[1])) {
        cout << "Testing the model" << endl;
        myCIC.test("input/TS1.txt","input/TS2.txt",
                "input/data.txt");
    } else {
        cerr << "ERROR, 1st positional argument: train or test" << endl;
        return 1;       
    };

    cout << endl;
    cout << "CIC ended" << endl;
    return 0;
}
//##########################################################################
