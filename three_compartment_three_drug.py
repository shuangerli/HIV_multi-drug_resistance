import numpy as np 
import matplotlib.pyplot as plt
import sys
import time
import os


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
	trajecotry = np.zeros((max_time, 8)) #number of WT, number of R1, heterozygosity
	trajecotry[0] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
	heterozygosity_record = np.zeros((max_time, 3))
	heterozygosity_record[0] = np.array([0.0, 0.0, 0.0])
	last_integer_time = 0
	valid = 1

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

		if int(time_record) > last_integer_time:
			#record abundance
			last_integer_time += 1
			trajecotry[last_integer_time] = compartmentA.all_n + compartmentB.all_n + compartmentC.all_n
			#record heterozygosity
			for i in range(len(compartments)):
				weight = compartments[i].n_total / max(n_all_compartments, 0.000001) #weighted by population size
				heterozygosity_record[last_integer_time] += compartments[i].calc_heterozygosity() * weight


		if n_all_compartments > equilibrium:
			containment_failure += 1

		#if time_record > 100:
			#print(n_all_compartments, time_record, compartmentA.all_n, compartmentB.all_n, compartmentC.all_n)

	if containment_failure == 0 and n_all_compartments > 0:
		print("didn't converge wt={}, r1={}".format(compartmentA.n_wt, compartmentA.n_r1))
		valid = 0 #discard this run

	return trajecotry, time_record, containment_failure >= 1, valid, heterozygosity_record



def main():
	n0 = int(sys.argv[1])
	s1 = float(sys.argv[2])
	s2 = float(sys.argv[3])
	s3 = float(sys.argv[4])
	p_mutation = float(sys.argv[5])
	p_migration = float(sys.argv[6])
	k = int(sys.argv[7])
	reps = int(sys.argv[8])
	seed = int(sys.argv[9])

	max_time = 20 * n0
	np.random.seed(seed)

	all_trajectory = np.zeros((max_time, 8))
	all_heterozygosity = np.zeros((max_time, 3))
	all_time_record = 0.0
	all_contaiment_failure = 0.0
	longest_simulation = 0.0
	valid_rep = 0
	equilibrium = k * 3 * (1.0 - 1.0 / (1.0 + s1 + s2 + s3)) #the equilibrium of birth and death, plus some term for noise
	print(equilibrium)

	clock_start = time.time()

	while valid_rep < reps:
		curr_trajecotry, curr_time_record, curr_containment_failure, curr_valid, heterozygosity_curr = simulate(s1, s2, s3, 
			n0, p_mutation, p_migration, k, equilibrium, max_time)
		if curr_valid:
			valid_rep += 1
			all_trajectory += curr_trajecotry
			all_time_record += curr_time_record
			all_contaiment_failure += curr_containment_failure
			all_heterozygosity += heterozygosity_curr
			if curr_time_record > longest_simulation:
				longest_simulation = curr_time_record
		#print(i)

	print("clock_time=", time.time() - clock_start)

	all_trajectory /= valid_rep
	all_time_record /= valid_rep
	all_contaiment_failure /= valid_rep
	all_heterozygosity /= valid_rep

	print("avg_time=", all_time_record)
	print("prob(failure)=", all_contaiment_failure)
	print("longest_sim=", longest_simulation)




	colors = ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0", "#f0027f", "#bf5b17", "#666666"]
	genotypes = ["WT", "R1", "R2", "R3", "R12", "R13", "R23", "R123"]
	#plot average trajectories
	fig, ax = plt.subplots(1, 2, figsize = (8, 4))
	plotx = np.arange(0, int(longest_simulation + 1))
	plotwt = all_trajectory[:, 0][:len(plotx)]
	plotr1 = all_trajectory[:, 1][:len(plotx)] + plotwt

	ax[0].plot(plotx, plotwt, linestyle = '-', label = "WT", color = colors[0])
	ax[0].fill_between(plotx, np.zeros(len(plotx)), plotwt, color = colors[0], alpha = 0.5)
	plot_mutant_last = plotwt

	for i in range(1, 8):
		plot_mutant = all_trajectory[:, i][:len(plotx)] + plot_mutant_last
		ax[0].plot(plotx, plot_mutant, linestyle = '-', label = genotypes[i], color = colors[i])
		ax[0].fill_between(plotx, plot_mutant_last, plot_mutant, color = colors[i], alpha = 0.5)
		plot_mutant_last = plot_mutant
	ax[0].set_xlabel("generation")
	ax[0].set_ylabel("average abundance")
	ax[0].legend()

	#Plot heterozygosity
	for i in range(3):
		ax[1].plot(plotx, all_heterozygosity[:, i][:len(plotx)], label = genotypes[i + 1], color = colors[i + 1])

	ax[1].set_xlabel("generation")
	ax[1].set_ylabel("average heterozygosity")
	ax[1].legend()
	fig.tight_layout()
	plt.show()

	return

if __name__ == '__main__':
	main()






