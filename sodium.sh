wget https://download.libsodium.org/libsodium/releases/libsodium-1.0.18-stable.tar.gz
tar xvzf libsodium-1.0.18-stable.tar.gz
cd libsodium-stable
./configure && make check 
sudo make install 
sudo ldconfig cd ../
