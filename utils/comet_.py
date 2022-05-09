import os
from comet_ml import Experiment
from dotenv import load_dotenv
import matplotlib.pyplot as plt
load_dotenv()

# Create an experiment with your api key
def create_experiment(tags=None):
    """ Crea el experimento con los tags pasados como variable y lo devuelve."""
    experiment = Experiment(
        api_key=os.getenv('COMET_API_KEY'),
        project_name="proyecto1-grupof",
        workspace=os.getenv('COMET_WORKSPACE'),
        auto_param_logging=False,
    )

    if tags:
        for tag in tags:
            experiment.add_tag(tag)

    return experiment


def log_experiment(experiment:Experiment, params=None, metrics=None, best_params=None):
    """ Logea tanto los parametros como las metricas. """
    if params!=None:
        experiment.log_parameters(params)
    if metrics!=None:
        experiment.log_metric("AMS", metrics)
    if best_params!=None:
        experiment.log_other("best_params", best_params)

    return True

def plot_cv_scores(cmt_exp, scores):
    '''
    Entrada:
        cmt_exp: experimento comet
        scores: cross-validation AMS scores
    '''
    ### Registro de Gráficas ###
    plt.figure()
    plt.plot(scores,'*-')
    plt.xlabel('fold')
    plt.ylabel('AMS')
    plt.title('AMS score')
    plt.grid()
    
    cmt_exp.log_figure(figure_name="CV_AMS" ,figure=plt)
   
    return