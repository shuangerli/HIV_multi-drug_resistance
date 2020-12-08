import numpy as np 
import matplotlib.pyplot as plt
import sys
import time
import os


class Compartment:
	def __init__(self, s1, n0, p_mutation, k):
		self.s1 = s1      #selection pressure by drug1
		self.s2 = 0.0     #no drug2 yet
		self.s3 = 0.0     #no drug3 yet
		self.n_total = n0 #initial total population size
		self.n_wt = n0    #everyone is wildtype at beginning
		self.n_r1 = 0     #mutant:resistant to drug1
		self.p_mutation = p_mutation    #rate mutation, wt -> r1, assume no recurrent mutation
		self.p_migration = 0.0  #rate migration, there is no migration here
		self.k = k        #carring capacity


	#Assume ensity-dependent birth, density-independent death
	def moran_birth_death(self):

		br_wt = (1.0 - self.s1) * (1 - self.n_total/self.k) * self.n_wt   #birth of WT, Density-dependent birth, WT experiences selection s1
		br_r1 = (1.0) * (1 - self.n_total/self.k) * self.n_r1        #birth of mutant r1, Density-dependent birth
		dr_wt = 1.0 * self.n_wt   #death of WT, density-independent
		dr_r1 = 1.0 * self.n_r1   #death of r1, density-independent


		#events: wt birth, wt death, r1 birth, r1 death
		wait_times = np.array([float('inf')] * 4) #will never happen if that population size is 0
		if self.n_wt > 0:
			wait_times[0] = np.random.exponential(1.0 / br_wt)
			wait_times[1] = np.random.exponential(1.0 / dr_wt)

		if self.n_r1 > 0:
			wait_times[2] = np.random.exponential(1.0 / br_r1)
			wait_times[3] = np.random.exponential(1.0 / dr_r1)

		event = np.argmin(wait_times)
		time_elapsed = np.min(wait_times)

		#Hard-code what happens the next
		if event == 0:
			#Mutate to r1 with p=p_mutation
			mutate = np.random.binomial(n = 1, p = self.p_mutation)
			if mutate:
				self.n_r1 += 1 #Reproduce an r1
			else:
				self.n_wt += 1 #Reproduce a WT
		elif event == 1:
			self.n_wt -= 1
		elif event == 2:
			self.n_r1 += 1
		elif event == 3:
			self.n_r1 -= 1
		else:
			print("???")

		self.n_total = self.n_wt + self.n_r1

		return time_elapsed


def simulate(s1, n0, p_mutation, k, equilibrium, max_time):
	compartmentA = Compartment(s1, n0, p_mutation, k)
	time_record = 0.0
	containment_failure = 0 #contained as long as n_all_compartment < equilibrium
	n_all_compartments = n0
	trajecotry = np.zeros((max_time, 3)) #number of WT, number of R1, heterozygosity
	trajecotry[0] = np.array([0, 0, 0])
	last_integer_time = 0
	valid = 1

	while n_all_compartments > 0 and time_record < max_time - 1:

		time_elapsed = compartmentA.moran_birth_death()
		time_record += time_elapsed

		if int(time_record) > last_integer_time:
			#record abundance
			last_integer_time += 1
			trajecotry[last_integer_time][0: 2] = np.array([compartmentA.n_wt, compartmentA.n_r1])
			#record heterozygosity
			pwt = compartmentA.n_wt / max(compartmentA.n_total, 0.0001) #handle the case it extincts
			trajecotry[last_integer_time][2] = pwt * (1.0 - pwt)

		n_all_compartments = compartmentA.n_total

		if n_all_compartments > equilibrium:
			containment_failure += 1

		#print(n_all_compartments, time_record, equilibrium, max_time)

	#if time_record >= max_time - 1:
	if containment_failure == 0 and n_all_compartments > 0:
		print("didn't converge wt={}, r1={}".format(compartmentA.n_wt, compartmentA.n_r1))
		valid = 0 #discard this run

	return trajecotry, time_record, containment_failure >= 1, valid



def main():
	n0 = int(sys.argv[1])
	s1 = float(sys.argv[2])
	p_mutation = float(sys.argv[3])
	k = int(sys.argv[4])
	reps = int(sys.argv[5])
	seed = int(sys.argv[6])

	max_time = 10 * n0
	np.random.seed(seed)

	all_trajectory = np.zeros((max_time, 3))
	all_time_record = 0.0
	all_contaiment_failure = 0.0
	longest_simulation = 0.0
	equilibrium = k * (1.0 - 1.0 / (1.0 + s1)) #the equilibrium of birth and death

	clock_start = time.time()

	for i in range(reps):
		curr_trajecotry, curr_time_record, curr_containment_failure, curr_valid = simulate(s1, n0, p_mutation, k, equilibrium, max_time)
		if curr_valid:
			all_trajectory += curr_trajecotry
			all_time_record += curr_time_record
			all_contaiment_failure += curr_containment_failure
			if curr_time_record > longest_simulation:
				longest_simulation = curr_time_record
		#print(i)

	print("clock_time=", time.time() - clock_start)

	all_trajectory /= reps
	all_time_record /= reps
	all_contaiment_failure /= reps

	print("avg_time=", all_time_record)
	print("prob(failure)=", all_contaiment_failure)
	print("longest_sim=", longest_simulation)

	#plot average trajectories
	fig, ax = plt.subplots(1, 2, figsize = (8, 4))
	plotx = np.arange(0, int(longest_simulation + 1))
	plotwt = all_trajectory[:, 0][:len(plotx)]
	plotr1 = all_trajectory[:, 1][:len(plotx)] + plotwt
	ploth = all_trajectory[:, 2][:len(plotx)]

	ax[0].plot(plotx, plotwt, linestyle = '-', label = "WT", color = "#7fc97f")
	ax[0].plot(plotx, plotr1, linestyle = '-', label = "Mutant R1", color = "#beaed4")
	ax[0].fill_between(plotx, np.zeros(len(plotx)), plotwt, color = "#7fc97f", alpha = 0.5)
	ax[0].fill_between(plotx, plotwt, plotr1, color = "#beaed4", alpha = 0.5)
	ax[0].set_xlabel("generation")
	ax[0].set_ylabel("average abundance")
	ax[0].legend()

	ax[1].plot(plotx, ploth, "b-", label = "heterozygosity")
	ax[1].set_xlabel("generation")
	ax[1].set_ylabel("average heterozygosity")
	ax[1].legend()
	plt.tight_layout()
	plt.show()

	return

if __name__ == '__main__':
	main()






