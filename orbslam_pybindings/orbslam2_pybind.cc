#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<string>
#include<System.h>
#include<MapPoint.h>
#include<KeyFrame.h>
#include<Frame.h>
#include<Map.h>
#include<opencv2/core/core.hpp>
#include<ndarray_converter.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace ORB_SLAM2;

PYBIND11_MODULE(orbslam2, m) {
    m.doc() = "Python bindings for OrbSlam2 classes.";

    NDArrayConverter::init_numpy();

    //py::class_<KeyFrame>(m, "KeyFrame")
    //  .def(py::init<>

    py::class_<KeyFrame>(m, "KeyFrame")
      .def(py::init<Frame &, Map*, KeyFrameDatabase*>(), "F"_a, "pMap"_a, "pKFDB"_a)
      .def("GetPose", &KeyFrame::GetPose)
      .def("GetPoseInverse", &KeyFrame::GetPoseInverse)
      .def("GetCameraCenter", &KeyFrame::GetCameraCenter)
      .def("GetStereoCenter", &KeyFrame::GetStereoCenter)
      .def("GetRotation", &KeyFrame::GetRotation)
      .def("GetTranslation", &KeyFrame::GetTranslation)
      .def("GetConnectedKeyFrames", &KeyFrame::GetConnectedKeyFrames)
      .def("GetVectorCovisibleKeyFrames", &KeyFrame::GetVectorCovisibleKeyFrames)
      .def("GetBestCovisibilityKeyFrames", &KeyFrame::GetBestCovisibilityKeyFrames, "N")
      .def("GetCovisiblesByWeight", &KeyFrame::GetCovisiblesByWeight, "w"_a)
      .def("GetWeight", &KeyFrame::GetWeight, "w"_a)
      .def("GetChilds", &KeyFrame::GetChilds)
      .def("GetParent", &KeyFrame::GetParent)
      .def("hasChild", &KeyFrame::hasChild)
      .def("GetLoopEdges", &KeyFrame::GetLoopEdges)
      .def("GetMapPoints", &KeyFrame::GetMapPoints)
      .def("GetMapPointMatches", &KeyFrame::GetMapPointMatches)
      .def("TrackedMapPoints", &KeyFrame::TrackedMapPoints, "minObs"_a)
      .def("GetMapPoint", &KeyFrame::GetMapPoint, "idx"_a)
      .def("GetFeaturesInArea", &KeyFrame::GetFeaturesInArea, "x"_a, "y"_a, "r"_a)
      .def("IsInImage", &KeyFrame::IsInImage, "x"_a, "y"_a)
      .def("ComputeSceneMedianDepth", &KeyFrame::ComputeSceneMedianDepth, "q"_a=2);

    py::class_<MapPoint>(m, "MapPoint")
      .def(py::init<const cv::Mat &, KeyFrame*, Map*>(), "Pos"_a, "pRefKF"_a, "pMap"_a)
      .def(py::init<const cv::Mat &, Map*, Frame*, const int &>(), "Pos"_a, "pMap"_a, "pFrame"_a, "idxF"_a)
      .def("GetWorldPos", &MapPoint::GetWorldPos)
      .def("GetNormal", &MapPoint::GetNormal)
      .def("GetReferenceKeyFrame", &MapPoint::GetReferenceKeyFrame)
      .def("GetObservations", &MapPoint::GetObservations)
      .def("GetIndexInKeyFrame", &MapPoint::GetIndexInKeyFrame, "pKF"_a)
      .def("IsInKeyFrame", &MapPoint::IsInKeyFrame, "pKF"_a)
      .def("isBad", &MapPoint::isBad)
      .def("GetReplaced", &MapPoint::GetReplaced)
      .def("GetFoundRatio", &MapPoint::GetFoundRatio)
      .def("GetFound", &MapPoint::GetFound)
      .def("GetDescriptor", &MapPoint::GetDescriptor)
      .def("GetMinDistanceInvariance", &MapPoint::GetMinDistanceInvariance)
      .def("GetMaxDistanceInvariance", &MapPoint::GetMaxDistanceInvariance);

/**   Functions for manipulation. Not needed currently.
      .def("IncreaseVisible", &MapPoint::IncreaseVisible, "n"_a=1)
      .def("IncreaseFound", &MapPoint::IncreaseFound, "n"_a=1)
      .def("Replace", &MapPoint::Replace, "pMP"_a)
      .def("SetBadFlag", &MapPoint::SetBadFlag)
      .def("AddObservation", &MapPoint::AddObservation, "pKF"_a, "idx"_a)
      .def("EraseObservation", &MapPoint::EraseObservation, "pKF"_a)
      .def("SetWorldPos", &MapPoint::SetWorldPos, "Pos"_a)
      .def("ComputeDistinctiveDescriptors", &MapPoint::ComputeDistinctiveDescriptors)
      .def("UpdateNormalAndDepth", &MapPoint::UpdateNormalAndDepth)
      .def("PredictScale", &MapPoint::PredictScale, "currentDist"_a, "pKF"_a)
      .def("PredictScale", &MapPoint::PredictScale, "currentDist"_a, "pF"_a)
**/

    py::class_<System> system(m, "System");

    system.def(py::init<const std::string &, const std::string &, const System::eSensor, const bool>(),
              "strVocFile"_a, "strSettingsFile"_a, "sensor"_a, "bUseViewer"_a=true)
            .def("TrackStereo", &System::TrackStereo,
              "Proccess the given stereo frame. Images must be synchronized and rectified. \
                Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale. \
                Returns the camera pose (empty if tracking fails).",
              "imLeft"_a, "imRight"_a, "timestamp"_a)
            .def("TrackRGBD", &System::TrackRGBD,
              "Process the given rgbd frame. Depthmap must be registered to the RGB frame. \
                Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale. \
                Input depthmap: Float (CV_32F). \
                Returns the camera pose (empty if tracking fails).",
              "im"_a, "depthmap"_a, "timestamp"_a)
            .def("TrackMonocular", &System::TrackMonocular,
              "Proccess the given monocular frame \
              Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale. \
              Returns the camera pose (empty if tracking fails).",
              "im"_a, "timestamp"_a)
            .def("ActivateLocalizationMode", &System::ActivateLocalizationMode,
              "This stops local mapping thread (map building) and performs only camera tracking.")
            .def("DeactivateLocalizationMode", &System::DeactivateLocalizationMode,
              "This resumes local mapping thread and performs SLAM again.")
            .def("MapChanged", &System::MapChanged,
              "Returns true if there have been a big map change (loop closure, global BA) \
              since last call to this function")
            .def("Reset", &System::Reset,
              "Reset the system (clear map)")
            .def("Shutdown", &System::Shutdown,
              "All threads will be requested to finish. \
              It waits until all threads have finished. \
              This function must be called before saving the trajectory.")
            .def("SaveTrajectoryTUM", &System::SaveTrajectoryTUM,
              "Save camera trajectory in the TUM RGB-D dataset format. \
              Only for stereo and RGB-D. This method does not work for monocular. \
              Call first Shutdown() \
              See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset",
              "filename"_a)
            .def("SaveKeyFrameTrajectoryTUM", &System::SaveKeyFrameTrajectoryTUM,
              "Save keyframe poses in the TUM RGB-D dataset format. \
              This method works for all sensor input. \
              Call first Shutdown() \
              See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset",
              "filename"_a)
            .def("SaveTrajectoryKITTI", &System::SaveTrajectoryKITTI,
              "Save camera trajectory in the KITTI dataset format. \
              Only for stereo and RGB-D. This method does not work for monocular. \
              Call first Shutdown() \
              See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php",
              "filename"_a)
            .def("GetTrackingState", &System::GetTrackingState)
            .def("GetTrackedMapPoints", &System::GetTrackedMapPoints)
            .def("GetTrackedKeyPointsUn", &System::GetTrackedKeyPointsUn);

    py::enum_<System::eSensor>(system, "eSensor")
        .value("MONOCULAR",System::eSensor::MONOCULAR)
        .value("STEREO",System::eSensor::STEREO)
        .value("RGBD",System::eSensor::RGBD)
        .export_values();

}
