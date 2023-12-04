# Secure Epigenetic Pacemake Implementation

We here provide the python libraries required for simulating the DO, CSP and MLE as used for the proof of concept implementation in our article. </br>
</br>
The file main.py contains an example usage for the secure protocol based on the example data referenced below. </br>
In the same file we provide cleartext implementations of the algorithm, both original as publised <a href="https://pubmed.ncbi.nlm.nih.gov/29979108/" target="_blank">here</a> by Snir et al and also based on our algorithm avoiding division operations during the calculation phase. </br>
</br>
If you wish to cite this implementation in your derived work, please use the following BibTeX entry: </br>

# Example data

Example data is adapted from <a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74193" target="_blank">GSE74193</a> as structured in this <a href="https://epigeneticpacemaker.readthedocs.io/en/latest/epm_tutorial/" target="_blank">EPM cleartext implementation</a> </br>

# Installation
```
git clone --recursive git@github.com:ASEC-lab/EPM-code.git
cd EPM-code
pip3 install -r requirements.txt
```



# Usage

```
python3 main.py <arguments>
```
Where arguments are:

 -h, --help            show help message and exit
 
  -n POLYNOMIAL, --polynomial POLYNOMIAL Polynomial modulus degree    
  
  -p PRIMES, --primes PRIMES Number of primes                   
  
  -c CORRELATION, --correlation CORRELATION Correlation percentage      
  
  -r ROUNDS, --rounds ROUNDS number of CEM rounds
  
  -b BITS, --bits BITS  number bits per prime
  
  -a, --auto_recrypt    allow auto recrypt upon low noise level
  
  -o, --orig_cleartext  run the original cleartext algorithm
  
  -d, --cleartext_no_division run the original cleartext algorithm with no division



  
  

