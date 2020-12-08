import numpy as np 
import matplotlib.pyplot as plt
import sys
import time
import os
import csv


class Compartment:
	def __init__(self, s1, s2, s3, n0, p_mutation, p_migration, k, label):
		self.n_total = n0 #initial total population size
		self.p_mutation = p_mutation    #rate mutation, gaining one resistance, assuming no recurrent mutation
		self.p_migration = p_migration  #rate migration, there is no migration here
		self.k = k        #carring capacity
		#correspond to each genotype: WT, R1, R2, R3, R12, R13, R23, R123
		self.all_s = np.array([0.0, s1, s2, s3, s1 + s2, s1+ s3, s2 + s3, s1 + s2 + s3])
		self.all_s = self.all_s - (s1 + s2 + s3)
		self.all_n = np.array([n0, 0, 0, 0, 0, 0, 0, 0])
		self.label = label


	#Assume ensity-dependent birth, density-independent death
	def moran_birth_death_sample(self):

		if self.n_total == 0:
			return 0, float('inf')

		br_all = (self.all_s + 1.0) * (1 - self.n_total / self.k) * self.all_n
		dr_all = 1.0 * self.all_n
		migrate_all = self.p_migration * self.all_n
		events_all = np.concatenate((br_all, dr_all, migrate_all))
		events_all[events_all == 0] = 0.000000001 #clean up the zeros
		#print(events_all)
		wait_times = np.random.exponential(1.0 / events_all)
		#print(wait_times)

		event = np.argmin(wait_times)
		time_elapsed = np.min(wait_times)

		return event, time_elapsed

	def moran_birth_death_execute(self, event):
		migrator, destination = None, None
		#Hard-code what happens the next
		if event <= 7:
			self.n_total += 1
			#Mutate to r1 with p=p_mutation
			mutate = np.random.binomial(n = 1, p = self.p_mutation)
			if mutate == 0:
				self.all_n[event] += 1 #no mutation
			else:
				if event == 0:
					new_mutation = np.argmax(np.random.multinomial(n = 1, pvals = [1.0/3.0, 1.0/3.0, 1.0/3.0]))
					self.all_n[1 + new_mutation] += 1 #WT -> one mutation

				elif event == 1:
					new_mutation = np.argmax(np.random.multinomial(n = 1, pvals = [0.5, 0.5, 0.0]))
					self.all_n[4 + new_mutation] += 1 #R1 -> R12 or R13

				elif event == 2:
					new_mutation = np.argmax(np.random.multinomial(n = 1, pvals = [0.5, 0.0, 0.5]))
					self.all_n[4 + new_mutation] += 1 #R2 -> R12 or R23

				elif event == 3:
					new_mutation = np.argmax(np.random.multinomial(n = 1, pvals = [0.0, 0.5, 0.5]))
					self.all_n[4 + new_mutation] += 1 #R3 -> R13 or R23

				else:
					self.all_n[7] += 1 #bi-mutant -> tri-mutant

		elif event <= 15:
			self.all_n[event - 8] -= 1
			self.n_total -= 1

		else:
			migrator = event - 16
			self.all_n[migrator] -= 1 #migrate out
			self.n_total -= 1

			destination_prob = [0.5, 0.5, 0.5]
			destination_prob[self.label] = 0.0
			destination = np.argmax(np.random.multinomial(n = 1, pvals = destination_prob))

		return migrator, destination

	def calc_heterozygosity(self):
		p_r1 = (self.all_n[1] + self.all_n[4] + self.all_n[5] + self.all_n[7]) / max(self.n_total, 0.00001)
		p_r2 = (self.all_n[2] + self.all_n[4] + self.all_n[6] + self.all_n[7]) / max(self.n_total, 0.00001)
		p_r3 = (self.all_n[3] + self.all_n[5] + self.all_n[6] + self.all_n[7]) / max(self.n_total, 0.00001)

		h = np.array([p_r1 * (1 - p_r1), p_r2 * (1 - p_r2), p_r3 * (1 - p_r3)])
		return h

	def migration_in(self, migrator):
		self.all_n[migrator] += 1
		self.n_total += 1

		return



def simulate(s1, s2, s3, n0, p_mutation, p_migration, k, equilibrium, max_time):
	compartmentA = Compartment(s1, 0, 0, n0, p_mutation, p_migration, k, 0)
	compartmentB = Compartment(0, 0, s3, n0, p_mutation, p_migration, k, 1)
	compartmentC = Compartment(0, s2, 0, n0, p_mutation, p_migration, k, 2)
	compartments = [compartmentA, compartmentB, compartmentC]

	time_record = 0.0
	containment_failure = 0 #contained as long as n_all_compartment < equilibrium
	n_all_compartments = n0 * 3
	#trajecotry = np.zeros((max_time, 8)) #number of WT, number of R1, heterozygosity
	#trajecotry[0] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
	#heterozygosity_record = np.zeros((max_time, 3))
	#heterozygosity_record[0] = np.array([0.0, 0.0, 0.0])
	#last_integer_time = 0
	valid = 1
	time_resistant = 0

	#while n_all_compartments > 0 and containment_failure == 0 and time_record < max_time - 1:
	while n_all_compartments > 0 and time_record < max_time - 1:

		wait_times = np.array([0.0, 0.0, 0.0])
		events = [0, 0, 0]
		for i in range(len(compartments)):
			events[i], wait_times[i] = compartments[i].moran_birth_death_sample()
		the_compartment, time_elapsed = np.argmin(wait_times), np.min(wait_times)
		time_record += time_elapsed
		migrator, destination = compartments[the_compartment].moran_birth_death_execute(events[the_compartment])

		if migrator != None:
			compartments[destination].migration_in(migrator)

		n_all_compartments = compartmentA.n_total + compartmentB.n_total + compartmentC.n_total

		'''
		if int(time_record) > last_integer_time:
			#record abundance
			last_integer_time += 1
			trajecotry[last_integer_time] = compartmentA.all_n + compartmentB.all_n + compartmentC.all_n
			#record heterozygosity
			for i in range(len(compartments)):
				weight = compartments[i].n_total / max(n_all_compartments, 0.000001) #weighted by population size
				heterozygosity_record[last_integer_time] += compartments[i].calc_heterozygosity() * weight
		'''
		if time_resistant == 0 and (compartmentA.all_n[-1] + compartmentB.all_n[-1] + compartmentC.all_n[-1] > 0):
			time_resistant = time_record

		if n_all_compartments > equilibrium:
			containment_failure += 1

		#if time_record > 100:
			#print(n_all_compartments, time_record, compartmentA.all_n, compartmentB.all_n, compartmentC.all_n)

	if containment_failure == 0 and n_all_compartments > 0:
		print("didn't converge wt={}, r1={}".format(compartmentA.n_wt, compartmentA.n_r1))
		valid = 0 #discard this run

	return time_record, containment_failure >= 1, valid, time_resistant



def main():
	n0 = int(sys.argv[1])
	s1 = float(sys.argv[2])
	s2 = float(sys.argv[3])
	s3 = float(sys.argv[4])
	p_mutation = float(sys.argv[5])
	p_migrations = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
	k = int(sys.argv[6])
	reps = int(sys.argv[7])
	seed = int(sys.argv[8])

	max_time = 50 * n0
	np.random.seed(seed)

	equilibrium = k * 3 * (1.0 - 1.0 / (1.0 + s1 + s2 + s3)) #the equilibrium of birth and death

	result_all_time_record = np.zeros(len(p_migrations))
	result_all_failure = np.zeros(len(p_migrations))
	result_all_time_resistant = np.zeros(len(p_migrations))

	clock_start = time.time()

	for m in range(len(p_migrations)):
		p_migration = p_migrations[m]
		all_time_record = 0.0
		all_contaiment_failure = 0.0
		all_time_resistant = 0.0
		longest_simulation = 0.0
		resistant_rep = 0
		for i in range(reps):
			curr_time_record, curr_containment_failure, curr_valid, curr_time_resistant = simulate(s1, s2, s3,
				n0, p_mutation, p_migration, k, equilibrium, max_time)
			if curr_valid:
				all_time_record += curr_time_record
				all_contaiment_failure += curr_containment_failure
				all_time_resistant += curr_time_resistant
				resistant_rep += (curr_time_resistant > 0)
				if curr_time_record > longest_simulation:
					longest_simulation = curr_time_record
		all_time_record /= reps
		all_contaiment_failure /= reps
		all_time_resistant /= resistant_rep
		result_all_time_record[m] = all_time_record
		result_all_failure[m] = all_contaiment_failure
		result_all_time_resistant[m] = all_time_resistant

	print("clock_time=", time.time() - clock_start)

	with open("three_three_mig_2.csv", "w", newline = '') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		writer.writerow(["mutation_rates"] + [str(elem) for elem in p_migrations])
		writer.writerow(["time_all"] + [str(elem) for elem in result_all_time_record])
		writer.writerow(["fail"] + [str(elem) for elem in result_all_failure])
		writer.writerow(["resist"] + [str(elem) for elem in result_all_time_resistant])

	#plot average trajectories
	fig, ax = plt.subplots(1, 2, figsize = (8, 4))

	ax[0].plot(p_migrations, result_all_time_record, "o", color = "#fb8072", label = "before extinction")
	ax[0].plot(p_migrations, result_all_time_resistant, "o", color = "#80b1d3", label = "acquiring resistance")
	ax[0].set_xlabel("migration rate")
	ax[0].set_ylabel("average generation")
	ax[0].legend()

	ax[1].plot(p_migrations, result_all_failure, "o", label = "P(containment failure)", color = "#8dd3c7")
	ax[1].set_xlabel("migration rate")
	ax[1].set_ylabel("P(containment failure)")
	ax[1].legend()

	plt.tight_layout()
	plt.show()

	return

if __name__ == '__main__':
	main()