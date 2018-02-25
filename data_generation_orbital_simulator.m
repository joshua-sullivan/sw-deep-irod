%% Initializing 
clear all
close all
clc
rng('default')
GM_EARTH = s3_constants('GM_EARTH');
R_EARTH = s3_constants('R_EARTH');
J2_EARTH = s3_constants('J2_EARTH');
Odss = 1.991063853e-7; % Sun-Synchronous precession rate
deg2rads = pi/180;
arcsec2rads = deg2rads/3600;
prop_step = 60;

eop        = struct2array(load('eop_c04_14.mat'));
lsdata     = struct2array(load('ls_table.mat'));
gmdl       = struct2array(load('ggm01s'));
kpap       = struct2array(load('kpapf107.mat'));
eop_mdl    = 0;

%% Setting up training set dimensions
dim_meas_batch = 20;
dim_train = 14;
t_meas = 60*randi([3, 10], [1, dim_train]);

dim_azel = 2;
dim_ao = 6;
dim_aa = 4;
dim_dt = 1;

dim_instance = dim_azel + dim_ao + dim_aa + dim_dt;
% X = zeros((dim_instance*dim_meas_batch), dim_train);

%% Noise sampling
% Observer absolute orbit determination (AOD)
AOD_pos_mean = 20;
AOD_vel_mean = 0.1;
AOD_pos_sigma = 5;
AOD_vel_sigma = 0.025;
ref_pos_noise_sigma = abs((AOD_pos_sigma*randn(dim_train, 1)) + AOD_pos_mean);
ref_vel_noise_sigma = abs((AOD_vel_sigma*randn(dim_train, 1)) + AOD_vel_mean);

% Az/El bearing angles
AzEl_mean = 10*arcsec2rads;
AzEl_sigma = 5*arcsec2rads;
azel_noise_sigma = abs((AzEl_sigma*randn(dim_train, 1)) + AzEl_mean);

% Observer absolute attitude determination (AAD)
AAD_offaxis_mean = 10*arcsec2rads;
AAD_offaxis_sigma = 2*arcsec2rads;
att_offaxis_noise_sigma = abs((AAD_offaxis_sigma*randn(dim_train, 1)) + AAD_offaxis_mean);
% att_offaxis_noise_sigma = 10*arcsec2rads;
AAD_boresight_mean = 30*arcsec2rads;
AAD_boresight_sigma = 5*arcsec2rads;
att_boresight_noise_sigma = abs((AAD_boresight_sigma*randn(dim_train, 1)) + AAD_boresight_mean);
% att_boresight_noise_sigma = 30*arcsec2rads;

R_rtn2c_nominal = [1 0 0; 0 0 1; 0 -1 0];

%% Initial feature sampling
% OBSERVER ABSOLUTE ORBIT
% Uniform distribution for perigee altitude
hp_min = 300e03;
hp_max = 1200e03;
hp = sample_uniform([1, dim_train], hp_max, hp_min);
% Uniform distribution for eccentricity
e_min = 0.001;
e_max = 0.01;
e = sample_uniform([1, dim_train], e_max, e_min);
% Semi-major axis
a = (hp + R_EARTH)./(1 - e);
% Uniform distribution for inclination
i_min = 5*deg2rads;
i_max = 100*deg2rads;
% i_min = 30*deg2rads;
% i_max = 30*deg2rads;
i = sample_uniform([1, dim_train], i_max, i_min);
% Uniform distribution for RAAN
O_min = 0;
O_max = 360*deg2rads;
% O_min = 60*deg2rads;
% O_max = 60*deg2rads;
O = sample_uniform([1, dim_train], O_max, O_min);
% Uniform distribution for argument of perigee
w_min = 0;
w_max = 360*deg2rads;
% w_min = 120*deg2rads;
% w_max = 120*deg2rads;
w = sample_uniform([1, dim_train], w_max, w_min);
% Uniform distribution for initial mean anomaly
M0_min = 0;
M0_max = 360*deg2rads;
% M0_min = 180*deg2rads;
% M0_max = 180*deg2rads;
M0 = sample_uniform([1, dim_train], M0_max, M0_min);

oe_osc_obsv_init = [a; e; i; O; w; M0];

% INITIAL RELATIVE ORBIT
ada_min = -200;
ada_max = 200;
ada = sample_uniform([1, dim_train], ada_max, ada_min);

adL_min = -50e03;
adL_max = -3e03;
adL = sample_uniform([1, dim_train], adL_max, adL_min);

adex_min = -1e03;
adex_max = 1e03;
adex = sample_uniform([1, dim_train], adex_max, adex_min);

adey_min = -1e03;
adey_max = 1e03;
adey = sample_uniform([1, dim_train], adey_max, adey_min);

adix_min = -1e03;
adix_max = 1e03;
adix = sample_uniform([1, dim_train], adix_max, adix_min);

adiy_min = -1e03;
adiy_max = 1e03;
adiy = sample_uniform([1, dim_train], adiy_max, adiy_min);

aroe_osc_init = [ada; adL; adex; adey; adix; adiy];
roe_osc_init = aroe_osc_init./a;
 
%% Force modeling and satellite properties
gmdl_degree = 20;
gmdl_order  = 20; 
apply_drag = 1;
drag_model = 1;
apply_SRP = 1;
apply_TBsun = 1;
apply_TBmoon = 1;
apply_relativity = 0;
apply_control = 0;
apply_empacc = 0;
apply_polarmot = 1;
pInt = 0;
pMdl = [gmdl_degree, gmdl_order, apply_drag, drag_model, ...
        apply_SRP, apply_TBsun, apply_TBmoon, apply_relativity, ...
        apply_control, apply_empacc, apply_polarmot];
           
obsv_mass = 154.4;
obsv_Ad = 1.3;
obsv_Asrp = 2.5;
obsv_Cd = 2.5;
obsv_Cr = 1.32;
pSat_obsv = [obsv_mass, obsv_Ad, obsv_Asrp, obsv_Cd, obsv_Cr];

targ_mass = 142.5;
targ_Ad = 0.38;
targ_Asrp = 0.55;
targ_Cd = 2.25;
targ_Cr = 1.2;
pSat_targ = [targ_mass, targ_Ad, targ_Asrp, targ_Cd, targ_Cr];

t_cal0 = [2017, 1, 1, 0, 0, 0]; % GPS Time start epoch
t_gps0 = s3_epoch_caltogps(t_cal0);


%% Feature simulation
obsv_data(dim_train, 1) = struct();
targ_data(dim_train, 1) = struct();
rel_data(dim_train, 1) = struct();

for ii = 1:dim_train
    obsv_data(ii, 1).true.oe_osc = [];
    obsv_data(ii, 1).true.x_eci = [];
    obsv_data(ii, 1).true.q_rtn2c = [];
    obsv_data(ii, 1).meas.oe_osc = [];
    obsv_data(ii, 1).meas.x_eci = [];
    obsv_data(ii, 1).meas.q_rtn2c = [];
    targ_data(ii, 1).true.oe_osc = [];
    targ_data(ii, 1).true.x_eci = [];
    rel_data(ii, 1).true.roe_osc = [];
    rel_data(ii, 1).true.aroe_osc = [];
    rel_data(ii, 1).true.dx_rtn = [];
    rel_data(ii, 1).true.dr_cam = [];
    rel_data(ii, 1).true.az_el = [];
    rel_data(ii, 1).meas.az_el = [];
end


sim_time = zeros(dim_train, 1);
tic
parfor ii = 1:dim_train
    prop_start = 0;
    prop_end = (dim_meas_batch-1)*t_meas(1, ii);
    t_prop = (prop_start : prop_step : prop_end)';
    dim_t = length(t_prop);
    
    ref_pos_noise = ref_pos_noise_sigma(ii, 1)*randn(3, dim_t);
    ref_vel_noise = ref_vel_noise_sigma(ii, 1)*randn(3, dim_t); 
    
    az_el_noise = azel_noise_sigma(ii, 1)*randn(2, dim_t);
    
    mod_idx = 0;
    for jj = 1:dim_t
        
%         fprintf('Train instance %i/%i\n', ii, dim_train);
%         fprintf('Progress %05.1f\n',jj/dim_t*100);
        
        % Updating pSim time vector 
        t_gps = t_gps0;
        t_gps(2) = t_gps0(2) + ((jj-1)*prop_step);
        t_cal = s3_epoch_gpstocal(t_gps);

        % Updating pAux vector for truth simulation
        pAux_obsv = [t_gps, pSat_obsv, pMdl, pInt];
        pAux_targ = [t_gps, pSat_targ, pMdl, pInt];
        
        if jj == 1
            
            obsv_data(ii, 1).true.oe_osc(:, jj) = wrap_oe(oe_osc_obsv_init(:, ii));
            obsv_data(ii, 1).true.x_eci(:, jj) = s3_state_keptocart(obsv_data(ii, 1).true.oe_osc(:, jj));
            obsv_data(ii, 1).true.q_rtn2c(:, jj) = s3_rotation2quaternion(R_rtn2c_nominal);
            
            obsv_data(ii, 1).meas.x_eci(:, jj) = obsv_data(ii, 1).true.x_eci(:, jj) + cat(1, ref_pos_noise(:, jj), ref_vel_noise(:, jj));
            obsv_data(ii, 1).meas.oe_osc(:, jj) = wrap_oe(s3_state_carttokep(obsv_data(ii, 1).meas.x_eci(:, jj)));
            JitterDCM = AttitudeNoiseMatrix(att_offaxis_noise_sigma(ii, 1), att_boresight_noise_sigma(ii, 1));
            obsv_data(ii, 1).meas.q_rtn2c(:, jj) = s3_rotation2quaternion(JitterDCM*R_rtn2c_nominal);
            
            targ_data(ii, 1).true.oe_osc(:, jj) = wrap_oe(oeref_and_qnsroe2oedep(obsv_data(ii, 1).true.oe_osc(:, jj), roe_osc_init(:, ii)));
            targ_data(ii, 1).true.x_eci(:, jj) = s3_state_keptocart(targ_data(ii, 1).true.oe_osc(:, jj));
            
            rel_data(ii, 1).true.roe_osc(:, jj) = oe2roe(obsv_data(ii, 1).true.oe_osc(:, jj), targ_data(ii, 1).true.oe_osc(:, jj));
            rel_data(ii, 1).true.aroe_osc(:, jj) = rel_data(ii, 1).true.roe_osc(:, jj) * obsv_data(ii, 1).true.oe_osc(1, jj);
            rel_data(ii, 1).true.dx_rtn(:, jj) = s3_state_ecitortn(obsv_data(ii, 1).true.x_eci(:, jj), targ_data(ii, 1).true.x_eci(:, jj));
            rel_data(ii, 1).true.dr_cam(:, jj) = R_rtn2c_nominal*rel_data(ii, 1).true.dx_rtn(1:3, jj);
            
            rel_data(ii, 1).true.az_el(:, jj) = pos_cam_rel2azel(rel_data(ii,1).true.dr_cam(:, jj));
            rel_data(ii, 1).meas.az_el(:, jj) = rel_data(ii, 1).true.az_el(:, jj) + az_el_noise(:, jj);
                       
        else
            
            prior_oe_osc_obsv = obsv_data(ii, 1).true.oe_osc(:, jj-1);
            [oe_osc_sat_temp, ~] = s3_prop_GVEearthorbit(prop_step, prior_oe_osc_obsv, zeros(3,1), ...
                gmdl, lsdata, eop, eop_mdl, kpap, pAux_obsv);
            obsv_data(ii, 1).true.oe_osc(:, jj) = wrap_oe(oe_osc_sat_temp);
            obsv_data(ii, 1).true.x_eci(:, jj) = s3_state_keptocart(obsv_data(ii, 1).true.oe_osc(:, jj));
            obsv_data(ii, 1).true.q_rtn2c(:, jj) = s3_rotation2quaternion(R_rtn2c_nominal);
            obsv_data(ii, 1).meas.x_eci(:, jj) = obsv_data(ii, 1).true.x_eci(:, jj) + cat(1, ref_pos_noise(:, jj), ref_vel_noise(:, jj));
            obsv_data(ii, 1).meas.oe_osc(:, jj) = wrap_oe(s3_state_carttokep(obsv_data(ii, 1).meas.x_eci(:, jj)));
            JitterDCM = AttitudeNoiseMatrix(att_offaxis_noise_sigma(ii, 1), att_boresight_noise_sigma(ii, 1));
            obsv_data(ii, 1).meas.q_rtn2c(:, jj) = s3_rotation2quaternion(JitterDCM*R_rtn2c_nominal);
            
            prior_oe_osc_targ = targ_data(ii, 1).true.oe_osc(:, jj-1);
            [oe_osc_sat_temp, ~] = s3_prop_GVEearthorbit(prop_step, prior_oe_osc_targ, zeros(3,1), ...
                gmdl, lsdata, eop, eop_mdl, kpap, pAux_targ);
            targ_data(ii, 1).true.oe_osc(:, jj) = wrap_oe(oe_osc_sat_temp);
            targ_data(ii, 1).true.x_eci(:, jj) = s3_state_keptocart(targ_data(ii, 1).true.oe_osc(:, jj));
            
            rel_data(ii, 1).true.roe_osc(:, jj) = oe2roe(obsv_data(ii, 1).true.oe_osc(:, jj), targ_data(ii, 1).true.oe_osc(:, jj));
            rel_data(ii, 1).true.aroe_osc(:, jj) = rel_data(ii, 1).true.roe_osc(:, jj) * obsv_data(ii, 1).true.oe_osc(1, jj);
            rel_data(ii, 1).true.dx_rtn(:, jj) = s3_state_ecitortn(obsv_data(ii, 1).true.x_eci(:, jj), targ_data(ii, 1).true.x_eci(:, jj));
            rel_data(ii, 1).true.dr_cam(:, jj) = R_rtn2c_nominal*rel_data(ii, 1).true.dx_rtn(1:3, jj);
            
            rel_data(ii, 1).true.az_el(:, jj) = pos_cam_rel2azel(rel_data(ii,1).true.dr_cam(:, jj));
            rel_data(ii, 1).meas.az_el(:, jj) = rel_data(ii, 1).true.az_el(:, jj) + az_el_noise(:, jj);
        end     
    end
    
end
fprintf('Run-time: %5.3f sec\n', toc)
X = [];
Y = [];
for ii = 1:dim_train
    
    prop_start_save = 0;
    prop_end_save = (dim_meas_batch-1)*t_meas(1, ii);
    t_prop_save = (prop_start_save : prop_step : prop_end_save)';
    dim_t_save = length(t_prop_save);
    
    Xj = [];
    for jj = 1:dim_t_save
        
        if mod(t_prop_save(jj), t_meas(1, ii)) == 0
                
            Xj_temp = [rel_data(ii, 1).meas.az_el(:, jj); obsv_data(ii, 1).meas.oe_osc(:, jj);...
                       obsv_data(ii, 1).meas.q_rtn2c(:, jj)];
            Xj = cat(1, Xj, Xj_temp);
        end
        
    end
    Xj = cat(1, Xj, t_meas(1, ii));
    X = cat(2, X, Xj);
    Y = cat(2, Y, rel_data(ii, 1).true.aroe_osc(:, end));
end
save('DNN_IROD_backup_data.mat')
save('DNN_IROD_train_data.mat', 'X', 'Y')
save('DNN_IROD_raw_data.mat', 'obsv_data', 'targ_data', 'rel_data')

%%
% for ii = 1:dim_train
%     figure
%     plot(rel_data(ii,1).meas.az_el(1,:)./deg2rads, rel_data(ii,1).meas.az_el(2,:)./deg2rads)
%     hold on
%     plot(rel_data(ii,1).true.az_el(1,:)./deg2rads, rel_data(ii,1).true.az_el(2,:)./deg2rads)
%     grid on
%     hold on
%     plot(rel_data(ii,1).true.az_el(1,1)./deg2rads, rel_data(ii,1).true.az_el(2,1)./deg2rads, 'Marker', 'o', 'MarkerSize', 4)
%     drawnow
% end

