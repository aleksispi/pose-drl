function rewards = get_reward(all_recon_errs, visited_cams, force_terminated)

% Load global config settings
global CONFIG

% Make sure all_recon_errs long enough
if numel(visited_cams) > 2 && numel(all_recon_errs) < numel(visited_cams)
    nbr_interpolate = numel(visited_cams) + 1 - numel(all_recon_errs);
    if nbr_interpolate > 0
        all_recon_errs = [all_recon_errs(1) * ones(1, nbr_interpolate), all_recon_errs(end)];
    end
end

% Initialize von Mises rewards.
% NaN added at the end, to prevent "learning" from the end of an active-view.
% We have one less action taken than the number of visited views, as
% a trajectory looks like "view-1 --> view-2 --> ... --> view-N", i.e.,
% the arrows are one less than #views, and arrows represent the actions
% taken. But due to caffe input formatting the reward-holder must have
% equally many entries as #views, hence we set the final=NaN, and our code
% is such that we ignore gradients from this NaN-reward.
rewards_mises = [zeros(numel(all_recon_errs(1 : end - 1)), 1); nan];

% Add penalties for same cam choices
[~, unique_idxs] = unique(visited_cams);
same_penalties = ones(size(rewards_mises)) * CONFIG.agent_same_cam_penalty;
same_penalties(unique_idxs) = 0;
same_penalties = circshift(same_penalties, [-1, 0]);
rewards_mises = rewards_mises + same_penalties;

% Now we simply need to propagate the final reward properly
end_reward = 1 - all_recon_errs(end) / all_recon_errs(1);

% Viewpoint selection reward
if numel(rewards_mises) > 1
    rewards_mises(end - 1) = rewards_mises(end - 1) + end_reward;
    rewards_mises(1 : end - 1) = discount_rewards(rewards_mises(1 : end - 1));
end

% Continue-to-next-active-view (done) action reward
if CONFIG.agent_nbr_actions > 0
	% In this case, with no auto-stop, dont want to learn done branch
	rewards_done = nan(size(rewards_mises));
else
    if CONFIG.agent_reward_autostop_future_improvement
        % If true the agent's done action recieves a reward when the action led
        % it to smaller reconstruction error in the future.
        improvement_rewards = zeros(size(rewards_mises));
        for i = 1 : numel(all_recon_errs) - 1
            future_best = min(all_recon_errs(i + 1: end));
            if future_best == 0
                improvement_rewards(i) = 0;
            else
                % Note: The below term differs a bit from eq. (8) in the paper,
                % but it has the same kind of effect
                improvement_rewards(i) = -(1 - all_recon_errs(i) / future_best);
            end
        end
        rewards_done = improvement_rewards + CONFIG.agent_step_penalty;
    else
        rewards_done = CONFIG.agent_step_penalty * ones(size(rewards_mises));
    end
    if force_terminated
        rewards_done(end) = rewards_done(end) + CONFIG.agent_force_stop_penalty;
    else
        rewards_done(end) = 0;
    end
    if ~CONFIG.agent_reward_autostop_future_improvement
        rewards_done(end) = rewards_done(end) + end_reward;
    end
    rewards_done = discount_rewards(rewards_done);
end

% Bundle rewards
rewards = struct('mises', rewards_mises(:), 'binary', rewards_done(:));
end

function discounted = discount_rewards(rewards)
    % Inner function for computing cumsum of rewards
    nan_idxs_rewards = isnan(rewards);
    rewards(nan_idxs_rewards) = 0;
    discounted = flipud(cumsum(flipud(rewards)));    
    discounted(nan_idxs_rewards) = nan;
end
