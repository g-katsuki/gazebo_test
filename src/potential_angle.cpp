#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <ryusei/common/defs.hpp>
#include <ryusei/navi/obstacle_detector.hpp>
#include <ryusei/navi/potential.hpp>
#include <string>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/pose_array.h>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <nav_msgs/msg/odometry.hpp>
#include <chrono>

#include <mutex>

namespace rs = project_ryusei;
using namespace std;
using namespace std::chrono_literals;

#define DEBUG 1  // to create potential map

namespace potential
{

/*** orientation から euler(roll, pitch, yaw)へ変換・保持するための構造体 ***/
struct RobotEuler {
  double roll_, pitch_, yaw_;

  RobotEuler(){
    roll_ = 0.; pitch_ = 0.; yaw_ = 0.;
  }

  RobotEuler(geometry_msgs::msg::Quaternion robot_orientation){
    update(robot_orientation);
  }

  void update(geometry_msgs::msg::Quaternion robot_orientation){
    tf2::Quaternion q(
      robot_orientation.x,
      robot_orientation.y,
      robot_orientation.z,
      robot_orientation.w
    );

    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw); 
    
    this->update(roll, pitch, yaw);
  }

  void update(double roll, double pitch, double yaw){
    roll_ = roll;
    pitch_ = pitch;
    yaw_ = yaw;
  };
};

class PotentialAngle : public rclcpp::Node
{
public:
  PotentialAngle(rclcpp::NodeOptions options);
  ~PotentialAngle();
private:
  rs::ObstacleDetector detector_;
  vector<rs::Obstacle> obstacles_;
  rs::Potential potential_;
  rs::Pose2D pose_;
  rs::Pose2D goal_;
  rs::Pose2D goal_abs_;
  rs::Pose2D goal_next_;
  std::vector<cv::Point3f> points_scan_;
  std::mutex mutex_;
  nav_msgs::msg::Odometry robot_odom_;
  nav_msgs::msg::Odometry robot_odom_ini_;
  double yaw_ini_;
  bool goal_flag_ = false;

  struct RobotEuler robot_euler_;
  std::unique_ptr<struct RobotEuler> robot_euler_ptr_;

  void onScanSubscribed(sensor_msgs::msg::LaserScan::SharedPtr msg);
  void onOdomSubscribed(nav_msgs::msg::Odometry::SharedPtr msg);
  void onGoalSubscribed(std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void on2GoalsSubscribed(std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void create_potential_map();
  rs::Pose2D relGoalToAbs(const nav_msgs::msg::Odometry odom_start, const nav_msgs::msg::Odometry odom_now,
                          const double yaw_start, const double yaw_now, const rs::Pose2D goal);

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_scan_;  // SharedPtrつけ忘れ注意
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_pose_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_goal_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_2goals_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_angle_;
};

/*** コンストラクタ ***/
PotentialAngle::PotentialAngle(rclcpp::NodeOptions options) : Node("potential_angle", options)
{
  using std::placeholders::_1;

  sub_scan_ = this->create_subscription<sensor_msgs::msg::LaserScan>("/scan", 10, 
                          std::bind(&PotentialAngle::onScanSubscribed, this, _1));
  sub_pose_ = this->create_subscription<nav_msgs::msg::Odometry>("/odom", 10,
                          std::bind(&PotentialAngle::onOdomSubscribed, this, _1));
  sub_goal_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/goal", 10,
                          std::bind(&PotentialAngle::onGoalSubscribed, this, _1));
  sub_2goals_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/goals", 10,
                          std::bind(&PotentialAngle::onGoalSubscribed, this, _1));
  pub_angle_ = this->create_publisher<std_msgs::msg::Float64>("/angle", 10);

  /*** ObstacleDetectorの設定ファイル読み込み ***/
  if(!detector_.init("data/navigation_cfg/navigation.ini"))
  {
    cout << "Failed to load config file" << endl;
  }

  /*** potentialの設定ファイル読み込み ***/
  if(!potential_.init("data/navigation_cfg/navigation.ini"))
  {
    cout << "Failed to load config file" << endl;
  }
}

/*** デストラクタ ***/
PotentialAngle::~PotentialAngle()
{

}

void PotentialAngle::onScanSubscribed(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  if(goal_flag_)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    sensor_msgs::msg::LaserScan latest_scan;
    cv::Point3f temp_point;
    points_scan_.clear();
    obstacles_.clear();
    latest_scan = *msg;

    int scan_size = latest_scan.ranges.size();

    /*** scanした点群をx,y座標に変換してvectorに格納 ***/
    for(int i = 0; i < scan_size; i++){
      string str = to_string(latest_scan.ranges[i]);
      if(!isdigit(str[0])) continue;
      double radian = ((double)i*360.0/(double)scan_size) * CV_PI/180.0;  // 左部はscanした角度の細かさに対しての調整
      temp_point.x = latest_scan.ranges[i] * sin(radian);
      temp_point.y = latest_scan.ranges[i] * cos(radian);
      temp_point.z = 1.0;
      points_scan_.push_back(temp_point);
    }

    detector_.visualizeLocalMapScan(points_scan_);
    detector_.detect(pose_, obstacles_);

    goal_abs_ = relGoalToAbs(robot_odom_ini_, robot_odom_, yaw_ini_ ,robot_euler_ptr_->yaw_, goal_);

    if((goal_abs_.x < 0.2 && goal_abs_.x > -0.2) && (goal_abs_.y < 0.2 && goal_abs_.y > -0.2))
    {
      goal_flag_ = false;
    }
    else
    {
      /*** 最適方向の計算 ***/
      double target_dir = potential_.findOptimalWay(pose_, obstacles_, goal_abs_);

      std_msgs::msg::Float64 target_pub;
      target_pub.data = target_dir;
      pub_angle_ -> publish(target_pub);

      /*** potential場を可視化するためにcsvファイルに保存 ***/
      #if DEBUG
      create_potential_map();
      #endif
      
      /*** ループ待機 ***/
      #if !DEBUG
      rclcpp::WallRate loop_rate(100ms);
      loop_rate.sleep();
      #endif
    }
  }
}

/*** Odomのorientationをyawに変換 ***/
void PotentialAngle::onOdomSubscribed(nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);

  robot_odom_.pose.pose.position    = msg->pose.pose.position;
  robot_odom_.pose.pose.orientation = msg->pose.pose.orientation;

  robot_euler_ptr_.reset(new struct RobotEuler(robot_odom_.pose.pose.orientation));
}

/*** goalを受信したら現在地点からgoalまでの位置関係を保持 ***/
void PotentialAngle::onGoalSubscribed(std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  vector<float> goal = msg->data;

  if(goal.size() != 3){
    cout << "Wrong Argumants: Usage is... [x(float),y(float),rad(float)]" << endl;
  }

  /*** goal設定時の状態を初期状態として保存 ***/
  robot_odom_ini_ = robot_odom_;
  yaw_ini_        = robot_euler_ptr_ -> yaw_;

  goal_ = rs::Pose2D(goal[0], goal[1], goal[2]);
  cout << "goal: " << goal_ << endl;

  goal_flag_ = true;
}

/*** 2goalsを受信したら現在地点からgoalsまでの位置関係を保持 ***/
void PotentialAngle::on2GoalsSubscribed(std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  vector<float> goal = msg->data;

  if(goal.size() != 6){
    cout << "Wrong Argumants: Usage is... [x(float),y(float),rad(float),x(float),y(float),rad(float)]" << endl;
  }

  /*** goal設定時の状態を初期状態として保存 ***/
  robot_odom_ini_ = robot_odom_;
  yaw_ini_        = robot_euler_ptr_ -> yaw_;

  goal_ = rs::Pose2D(goal[0], goal[1], goal[2]);
  goal_next_ = rs::Pose2D(goal[3], goal[4], goal[5]);

  goal_flag_ = true;
}

/*** ロボットからのゴールへの絶対座標を計算 ***/
rs::Pose2D PotentialAngle::relGoalToAbs(const nav_msgs::msg::Odometry odom_start, const nav_msgs::msg::Odometry odom_now,
                                        const double yaw_start, const double yaw_now, const rs::Pose2D goal)
{
  rs::Pose2D goal_abs;
  /*** 相対座標からみたgoalへのxとyの距離 ***/
  double dis_to_goal_x = goal.x - (odom_now.pose.pose.position.x - odom_start.pose.pose.position.x);
  double dis_to_goal_y = goal.y - (odom_now.pose.pose.position.y - odom_start.pose.pose.position.y);

  /*** 相対位置でのロボット(ロボットは0度とされる)とgoalの角度 ***/
  double angle_integrated = std::atan2(dis_to_goal_y, dis_to_goal_x);

  /*** ロボットの角度変化 ***/
  double angle_robot = yaw_now - yaw_start;

  /*** ロボットからみたgoalへの角度 ***/
  double angle_to_goal = angle_integrated - angle_robot;

  /*** 直線距離(r)から、ロボットからgoalへのx,yの距離を算出 ***/
  double r = sqrt(dis_to_goal_x * dis_to_goal_x + dis_to_goal_y * dis_to_goal_y);
  goal_abs.x = r * cos(angle_to_goal);
  goal_abs.y = r * sin(angle_to_goal);

  cout << "x to goal: " << goal_abs.x << endl;
  cout << "y to goal: " << goal_abs.y << endl;

  return goal_abs;
}

/*** potential場を可視化するためにcsvファイルに保存 ***/
void PotentialAngle::create_potential_map()
{
  /*** 計算したポテンシャルをファイルに保存 ***/
  cv::Mat potential_map;
  potential_.createPotentialMap(pose_, obstacles_, goal_abs_, detector_.RANGE_X_,
                                detector_.RANGE_Y_, detector_.UNIT_, potential_map);
  ofstream ofs("data/potential.csv");
  /*** ヘッダ部の書き込み ***/
  ofs << "RangeX," << detector_.RANGE_X_ << endl;
  ofs << "RangeY," << detector_.RANGE_Y_ << endl;
  ofs << "Unit," << detector_.UNIT_ << endl << endl;
  /*** ポテンシャル場の書き込み ***/
  for(int i = 0; i < potential_map.rows; i++){
    for(int j = 0; j < potential_map.cols; j++){
      float val = potential_map.at<float>(i, j);
      if(j == potential_map.cols - 1) ofs << val;
      else ofs << val << ",";
    }
    if(i < potential_map.rows - 1) ofs << endl;
  }
  ofs.close();
}

}
/*** PotentialAngleクラスをコンポーネントとして登録 ***/
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(potential::PotentialAngle)