Rem activate environment
cd thesis_env/Scripts/
call activate.bat
echo "Environment activated"
cd ../..

Rem install requirements
cd deep-spectral-segmentation
pip install -r requirements.txt
echo "Requirements installed"

Rem move to working directory
cd extract