function run_train_agent(train_mode)
% Script for executing training of Pose-DRL

% Clean sheet
clc;
close all;

if ~exist('train_mode', 'var')
    train_mode = 'train';
end

% Also allow numeric inputs
if train_mode == 0
	train_mode = 'train';
elseif train_mode == 1
	train_mode = 'val';
end

% Add paths
addpath(genpath('code/'));

% Setup global config settings
load_config(train_mode)

% Launch training
train_agent();
