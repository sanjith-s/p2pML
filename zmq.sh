git clone http://github.com/zeromq/libzmq.git 
cd libzmq 
./autogen.sh 
./configure -with-libsodium && make 
sudo make install 
sudo ldconfig cd ../
