"""
In this file, we apply the Grounhog attack to the Census 1% Teaching file of
the England and Wales census and the Raw generator.

The goal of PrivE is to evaluate the possibility of an attack, rather than
train and deploy attacks against real datasets. This informs the design
decisions made, especially as relates to the auxiliary knowledge.

"""

# This is only required if you haven't installed PrivE as a module.
import sys

sys.path.append("..")

import prive
import prive.datasets
import prive.generators
import prive.threat_models
import prive.attacks
import prive.report

from sklearn.ensemble import RandomForestClassifier

# We attack the 1% Census Microdata file, available at:
#  https://www.ons.gov.uk/census/2011census/2011censusdata/censusmicrodata/microdatateachingfile
# We have created a .json description file, so that prive.Dataset.read can load both.
data = prive.datasets.TabularDataset.read(
    "data/2011 Census Microdata Teaching File"
)

# We attack the (trivial) Raw generator, which outputs its training dataset.
generator = prive.generators.Raw()

# We now define the threat model: what is assumed that the attacker knows.
# We first define what the attacker knows about the dataset. Here, we assume
# they have access to an auxiliary dataset from the same distribution.
data_knowledge = prive.threat_models.AuxiliaryDataKnowledge(
    # The attacker has access to 50% of the data as auxiliary information.
    # This information will be used to generate training datasets.
    data,
    sample_real_frac=0.5,
    # The attacker knows that the real dataset contains 5000 samples. This thus
    # reflects the attacker's knowledge about the real data.
    num_training_records=5000,
)

# We then define what the attacker knows about the synthetic data generator.
# This would typically be black-box knowledge, where they are able to run the
# (exact) SDG model on any dataset that they choose, but can only observe
# (input, output) pairs and not internal parameters.
sdg_knowledge = prive.threat_models.BlackBoxKnowledge(
    generator,
    # The attacker also specifies the size of the output dataset. In practice,
    # use the size of the published synthetic dataset.
    num_synthetic_records=5000,
)

# Now that we have defined the attacker's knowledge, we define their goal.
# We will here focus on a membership inference attack on a random record.
threat_model = prive.threat_models.TargetedMIA(
    attacker_knowledge_data=data_knowledge,
    # We here select the first record, arbitrarily.
    target_record=data.get_records([0]),
    attacker_knowledge_generator=sdg_knowledge,
    # These are mostly technical questions. They inform how the attacker will
    # be trained, but are not impactful changes of the threat model.
    #  - do we generate pairs (D, D U {target}) to train the attack?
    generate_pairs=True,
    #  - do we append the target to the dataset, or replace a record by it?
    replace_target=True,
)

# Next step: initialise an attacker. Here, we just apply the GroundHog attack
# with standard parameters (from Stadler et al., 2022).
attacker = prive.attacks.GroundhogAttack(
	# The GroundhogAttack attacker is mostly a wrapper over a set classifier.
	# We here use, as in Stadler et al., a feature-based set classifier, which
	#  (1) computes a vector of (fixed) features of the set to classify.
	#  (2) trains a vector-based classifier using these features.
	prive.attacks.FeatureBasedSetClassifier(
		# We use the F_naive and F_hist fatures (from the paper).
		prive.attacks.NaiveSetFeature() + prive.attacks.HistSetFeature(),
		# We use a random forest with 100 trees and default parameters.
		RandomForestClassifier(n_estimators = 100)
	)
)

# Having defined all the objects that we need, we can train the attack.
attacker.train(
	# The TargetedMIA threat model is a TrainableThreatModel: it defines a method
	#  to generate training samples (synthetic_dataset, target_in_real_dataset).
	# This is why the threat model is passed to train the attacker.
	threat_model,
	# This is the number of training pairs generated by the threat model to
	# train the attacker.
	num_samples = 1000,
)

# The attack is trained! Evaluate it within the test model.
# [explain why we split this way.]
attack_labels, truth_labels = threat_model.test(attacker, num_samples=1000)

# Finally, generate a report to evaluate the results.
attack_summary = prive.report.MIAttackSummary(
	# The summary requires these
	attack_labels, truth_labels,
	# And some metadata for nicer displays.
	generator_info = 'raw',
	attack_info = attacker.__name__,
	dataset_info = 'Census',
	target_id = '1',
)

# Output nice, printable metrics that evaluate the attack.
metrics = attack_summary.get_metrics()
print('Results:', metrics)