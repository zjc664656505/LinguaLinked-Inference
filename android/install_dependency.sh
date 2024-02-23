#!/bin/bash
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y net-tools
sudo apt-get install -y git
sudo apt-get install -y  iperf3
sudo apt-get install -y  python-is-python3
sudo apt-get install -y pip
sudo apt-get install -y inetutils-ping
sudo apt-get install -y vim
sudo apt-get install -y ssh
pip install torch torchvision numpy

sshd
