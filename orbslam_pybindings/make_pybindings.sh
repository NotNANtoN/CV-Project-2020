cp ./additional_build_files/CMakeLists.txt ../ORB_SLAM2/
cp ./additional_build_files/System.h ../ORB_SLAM2/include/
cp ./additional_build_files/LoopClosing.h ../ORB_SLAM2/include/
cp ./additional_build_files/Tracking.h ../ORB_SLAM2/include/
cp ./additional_build_files/Initializer.cc ../ORB_SLAM2/src/
cp ./additional_build_files/System.cc ../ORB_SLAM2/src/
cp ./additional_build_files/Tracking.cc ../ORB_SLAM2/src/
mkdir ../ORB_SLAM2/build
cd ../ORB_SLAM2
chmod +x ./build.sh
./build.sh
