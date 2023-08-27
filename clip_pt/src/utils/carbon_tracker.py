import torch
import wandb
from codecarbon import EmissionsTracker


def carbon_tracker_init(tracking_mode, gpu_ids, carbon_checker):
    if carbon_checker == 'true':
        tracker_code_carbon = EmissionsTracker(log_level='error', tracking_mode=tracking_mode,
                                               gpu_ids=gpu_ids)
        tracker_code_carbon.start()
    else:
        tracker_code_carbon = None
        print('Cuda Version: ' + str(torch.version.cuda))
        print('Carbon Tracke Disabled by the user.')
    return tracker_code_carbon


def carbon_tracker_end(tracker_code_carbon, country_carbon, carbon_checker):
    if carbon_checker == 'true':
        print("Stopping the carbon checker")
        our_emission = tracker_code_carbon.stop()
        print("Calculating Carbon and Energy Expenditure")
        energy_in_kwh = tracker_code_carbon.final_emissions_data.energy_consumed
        print("Logging in wandb")
        wandb.log({"carbon/Final Emission (CodeCarbon)": our_emission})
        wandb.log({"carbon/Final Emission": energy_in_kwh * country_carbon})
        wandb.log({"carbon/Final Energy": energy_in_kwh})
        return our_emission, energy_in_kwh
    else:
        return 0,0