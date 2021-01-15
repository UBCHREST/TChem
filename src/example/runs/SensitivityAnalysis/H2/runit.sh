
exec=$TCHEM_INSTALL_PATH/example/TChem_SensitivityAnalysisSourceTerm.x
inputs=inputs/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
inputfile=$inputs"sample.dat"
outputfile="test.dat"

$exec --outputfile=$outputfile --chemfile=$chemfile --thermfile=$thermfile --inputfile=$inputfile
