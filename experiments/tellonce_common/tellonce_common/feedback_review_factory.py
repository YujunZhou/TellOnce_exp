"""Create the simulated-user feedback agent for a training worker."""
from .feedback_transport import FeedbackAgent


def make_feedback_agent(spec, roles, *, phase, record):
    if phase != 'training' or spec.identity != roles.identity or spec.binding != roles.binding:
        raise ValueError('simulated-user feedback belongs to the training owner')
    return FeedbackAgent(roles.get('feedback'))
