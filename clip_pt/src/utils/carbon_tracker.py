import torch
import wandb
from codecarbon import EmissionsTracker

def carbon_tracker_init(tracking_mode, gpu_ids):
	if float(torch.version.cuda) >= 12:
		tracker_code_carbon = EmissionsTracker(log_level = 'error', tracking_mode=tracking_mode, gpu_ids=gpu_ids)
		tracker_code_carbon.start()
	else:
		tracker_code_carbon = None
		print('Cuda Version: ' + str(torch.version.cuda))
		print('Carbon Tracke Disabled')
	return tracker_code_carbon

def carbon_tracker_end(tracker_code_carbon):
    if float(torch.version.cuda) >= 12:
        our_emission = tracker_code_carbon.stop()
        energy_in_kwh = tracker_code_carbon.final_emissions_data.energy_consumed
        wandb.log({"carbon/Final Emission (CodeCarbon)": our_emission})
        wandb.log({"carbon/Final Emission": energy_in_kwh * config.carbon["brazil_carbon_intensity"]})
        wandb.log({"carbon/Final Energy": energy_in_kwh})
    	return our_emission, energy_in_kwh
