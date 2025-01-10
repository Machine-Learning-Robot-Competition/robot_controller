# Robot Controller

This is a ROS package designed and created by Felix Toft and Joshua Riefman for the Fall 2024 ENPH353 autonomous machine learning robot competition. This package consists of all of the nodes, UI, and TensorFlow models used to control our robot during competition in which we achieved a nearly flawless score.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Nodes](#nodes)

## Overview

The competition took place inside of ROS Gazebo. The challenge was to build an entirely autonomous robot to proceed through a simulated environment (in Gazebo) to solve a murder. As the robot progressed through the environment on its own, it would need to identify and read clues off of boards, whilst avoiding obstacles and not getting lost. This year, teams were allowed to create their own robots instead of using a default car: we decided to use a drone and fly.

For our robot, we adapted the Hector Quadrotor open-source ROS package(s) developed by Team Hector at the Technical University of Darmstadt in Germany, which researches and produces search and rescue robots. Specifically, we adapted their drone control and physics suite, but we injected our own sensors, models, and developed our own localization and navigation suite.  

![Drone](/media/drone.png)

## Architecture
At a high level, we used visual SLAM for localization (using absolute coordinates was forbbiden was it trivialized the challenge of the competition) using a pre-trained map and had the drone fly from clueboard to clueboard and read clues by feeding images through a letter extraction pipeline which fed into a convolutional neural network for letter identification.

We can express the high-level structure of our robot’s control as a block diagram.

![Software Architecture Block Diagram](/media/software_diagram.png)

Our Robot UI node, while having the initial purpose of displaying a helpful interface from which we could debug and observe the robot’s progress, had its role expanded into a high-level executor, managing the launch and lifetime of nearly all of the ROS nodes used. This served a single purpose and executed it perfectly: all logs went to the same place, meaning only one terminal window must be observed to see everything going on.

## Nodes
Our design philosophy was to have rather granular nodes which have a single purpose. 

1. `Master`: An additional layer of high-level orchestration was accomplished with a Master node, with a rather limited but critical role. Simply, the master node orchestrated the process of going to a clueboard, telling the Brain node to read it when we should be able to see it properly, then going to the next clueboard, and so on. 
2. `Navigation`: Based on the Master node’s command, the Navigation node uses PID in a control loop to navigate to a waypoint. 
3. `Control`: The Control node fuses an altitude request from the Master node and a velocity request from the Navigation node into a single Twist which it publishes to the Hector Quadrotor physics suite. 
4. `Brain`: The Brain node, upon receiving the command to try to read a clueboard from the Master node, will extract the clueboard, extract the letters, recognize the letters, and finally send the read clue to the score tracker. 
5. `Odometry`: The odometry node’s job was to subscribe to velocity sensor topics and estimate position based on velocity data. 
6. `SLAM`: The SLAM node was launched at startup from a launch file, and not managed by the UI, and would handle map generation or loading, localization, and transform publishing. 
