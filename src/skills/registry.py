# Import skill classes
from .restaurant_skill import RestaurantSkill
from .realestate_skill import RealEstateSkill

# Map skill IDs (from config) to skill classes
SKILL_REGISTRY = {
    RestaurantSkill.skill_id: RestaurantSkill,
    RealEstateSkill.skill_id: RealEstateSkill,
}

def get_skill_class(skill_id):
    """Gets the skill class from the registry based on ID."""
    skill_cls = SKILL_REGISTRY.get(skill_id)
    if not skill_cls:
        raise ValueError(f"Unknown skill ID '{skill_id}' configured.")
    return skill_cls