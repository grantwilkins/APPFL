# Install dependencies for APPFL
git clone https://github.com/grantwilkins/APPFL.git
cd APPFL
git checkout sz-compression
sudo apt-get install python3-pip python3 
pip3 install -e ".[dev,examples,analytics]"
pip3 install zfpy scipy

# Install lossy compression dependencies
