from qiskit import QuantumRegister, QuantumCircuit, BasicAer, execute
from qiskit.aqua.components.uncertainty_models import GaussianConditionalIndependenceModel
from qiskit.aqua.components.uncertainty_problems import UnivariatePiecewiseLinearObjective
from qiskit.aqua.components.uncertainty_problems import MultivariateProblem
from qiskit.aqua.circuits import WeightedSumOperator
from qiskit.aqua.circuits  import FixedValueComparator
from qiskit.aqua.algorithms import AmplitudeEstimation
from bisection_search import * 
import numpy as np
import matplotlib.pyplot as plt



#partie 0: calcul de VaR, Loss et CVar avec le modele d'incertitude sans utiliser le QAE
print("calcul de VaR, Loss et CVar avec le modele d'incertitude sans utiliser le QAE")

# definition du backend
backend = BasicAer.get_backend('statevector_simulator') 

# parametres du problème

#nombre de qubits utilisés pour représenter Z, où Z est une variable aleatoire modelisant l'evolution du portfeuil
n_z = 2
#valeur de la troncature pour Z
z_max = 2
#Valeurs possibles de Z. Z peut prendre 2**n_z valeurs possibles entre -z_max et +z_max
z_values = np.linspace(-z_max, z_max, 2**n_z)
#la probabilité par default de chaque asset
p_zeros = [0.15, 0.25]
#sensibilités de probabilités par default
rhos = [0.1, 0.05]
#perte par défaut pour l'actif k
lgd = [1, 2]

K = len(p_zeros)
#niveau de confiance pour la VaR
alpha = 0.05

# construire le circuit pour le modèle d'incertitude (Gaussian Conditional Independence model)
u = GaussianConditionalIndependenceModel(n_z, z_max, p_zeros, rhos)


# déterminer le nombre de qubits requis pour représenter le modèle d'incertitude
num_qubits = u.num_target_qubits

# initialiser le registre quantique et le circuit quantique
q = QuantumRegister(num_qubits, name='q')
qc = QuantumCircuit(q)


#print(qc.draw())

#construire le circuit : Ajoute le sous-circuit correspondant au circuit donné
u.build(qc, q)

# exécuter la simulation
job = execute(qc, backend=BasicAer.get_backend('statevector_simulator'))


# analyze uncertainty circuit and determine exact solutions
p_z = np.zeros(2**n_z)  #tableau représentant la probabilité de chaque valeur possible de Z (chaque position est initialisée à 0%)
p_default = np.zeros(K) #tableau représentant la probabilité par default pour z égal a zero
values = []				#initializer le tableau qui contient la perte pour chaque asset
probabilities = []		#la probabilité qui correspond à chaque valeur de Z

#print(job.result().get_statevector())

#boucle pour parcourrir tous les états possibiles pour notre circuit
for i, a in enumerate(job.result().get_statevector()):
    
    # get binary representation
    b = ('{0:0%sb}' % num_qubits).format(i)
    prob = np.abs(a)**2 	#prende la probabilite d'etre dans l'etat z. on a pris alpha au carré car beta au carré est negligeable 

    # au lieu d'avoir 16 états differentes l'un à l'autre on fait de telle sort qu'on a 4 états que se repetent    
    i_normal = int(b[-n_z:], 2)
    p_z[i_normal] += prob

    # creation d'un vector qui contient la perte de chaque asset
    loss = 0
    for k in range(K):
        if b[K - k - 1] == '1':
            p_default[k] += prob
            loss += lgd[k]

    values += [loss]
    
    probabilities += [prob]  


values = np.array(values)
probabilities = np.array(probabilities)
# print("probabilities: ",probabilities)

#calcul de la perte total ou expected_loss
expected_loss = np.dot(values, probabilities)
# print("expected_loss: ",expected_loss)

#trier le vector de pertes
losses = np.sort(np.unique(values)) 
# print("losses: ",losses)

pdf = np.zeros(len(losses)) #initialization du vector avec la probabilité de chaque possibilité de perte

for i, v in enumerate(losses):
    pdf[i] += sum(probabilities[values == v]) #mis à jour du PDF

#calcul de la probabilité cumulé de chaque possibilité de perte
cdf = np.cumsum(pdf) 
# print("pdf: ", pdf) 
# print("cdf: ", cdf )

#calcul de VaR et CVaR
i_var = np.argmax(cdf >= 1-alpha)
exact_var = losses[i_var]
exact_cvar = np.dot(pdf[(i_var+1):], losses[(i_var+1):])/sum(pdf[(i_var+1):])


print('Expected Loss E[L]:                %.4f' % expected_loss) 
print('Value at Risk VaR[L]:              %.4f' % exact_var) 
print('P[L <= VaR[L]]:                    %.4f' % cdf[exact_var]) 
print('Conditional Value at Risk CVaR[L]: %.4f' % exact_cvar)


# # tracer PDF, expected loss, var et cvar
# plt.bar(losses, pdf)
# plt.axvline(expected_loss, color='green', linestyle='--', label='E[L]')
# plt.axvline(exact_var, color='orange', linestyle='--', label='VaR(L)')
# plt.axvline(exact_cvar, color='red', linestyle='--', label='CVaR(L)')
# plt.legend(fontsize=15)
# plt.xlabel('Loss L ($)', size=15)
# plt.ylabel('probability (%)', size=15)
# plt.title('Loss Distribution', size=20)
# plt.xticks(size=15)
# plt.yticks(size=15)
# plt.show()

# # tracer les results pour Z
# plt.plot(z_values, p_z, 'o-', linewidth=3, markersize=8)
# plt.grid()
# plt.xlabel('Z value', size=15)
# plt.ylabel('probability (%)', size=15)
# plt.title('Z Distribution', size=20)
# plt.xticks(size=15)
# plt.yticks(size=15)
# plt.show()

# # tracer les results pour la probabilité par default
# plt.bar(range(K), p_default)
# plt.xlabel('Asset', size=15)
# plt.ylabel('probability (%)', size=15)
# plt.title('Individual Default Probabilities', size=20)
# plt.xticks(range(K), size=15)
# plt.yticks(size=15)
# plt.grid()
# plt.show()




#partie 2 estimation de la perte en utilisant QAE
print("calcul de VaR, Loss et CVar avec le QAE")
print("----------estimation de Expected Loss ------------")

# déterminer le nombre de qubits requis pour représenter la perte totale
n_s = WeightedSumOperator.get_required_sum_qubits(lgd)

# construire le circuit pour realiser la somme pondérée
agg = WeightedSumOperator(n_z + K, [0]*n_z + lgd)


# définition de la fonction Objective
breakpoints = [0]
slopes = [1]
offsets = [0]
f_min = 0
f_max = sum(lgd)
c_approx = 0.25

objective = UnivariatePiecewiseLinearObjective(
    agg.num_sum_qubits,
    0,
    2**agg.num_sum_qubits-1,  # valeur max qui peut être atteinte par les qubits (ne sera pas toujours atteinte)
    breakpoints, 
    slopes, 
    offsets, 
    f_min, 
    f_max, 
    c_approx
)

# define overall multivariate problem
multivariate = MultivariateProblem(u, agg, objective)

#trouver le nombre de qubit nécessaires pour le MultivariateProblem
num_qubits = multivariate.num_target_qubits
#trouver le nombre de qubits auxilliaires pour notre problème
num_ancillas = multivariate.required_ancillas()

#construction des registres et du circuit
q = QuantumRegister(num_qubits, name='q')
q_a = QuantumRegister(num_ancillas, name='q_a')
qc = QuantumCircuit(q, q_a)
multivariate.build(qc, q, q_a)


# executer l'estimation
num_eval_qubits = 5
ae = AmplitudeEstimation(num_eval_qubits, multivariate)

#executer le QAE pour estimer la perte
result = ae.run(quantum_instance=BasicAer.get_backend('statevector_simulator'))


print('Exact Expected Loss E[L]:     \t%.4f' % expected_loss)
print('Estimated Expected Loss E[L]: \t%.4f' % result['estimation'])
print('probability:                  \t%.4f' % result['max_probability'])



# # tracer les valeurs estimées pour "a".
# plt.bar(result['values'], result['probabilities'], width=0.5/len(result['probabilities']))
# plt.xticks([0, 0.25, 0.5, 0.75, 1], size=15)
# plt.yticks([0, 0.25, 0.5, 0.75, 1], size=15)
# plt.title('"a" Value', size=15)
# plt.ylabel('Probability', size=15)
# plt.ylim((0,1))
# plt.grid()
# plt.show()

# tracer les valeurs estimées de la perte attendue (après redimensionnement et inversion de la transformation c_approx)
# plt.bar(result['mapped_values'], result['probabilities'], width=1/len(result['probabilities']))
# plt.axvline(expected_loss, color='red', linestyle='--', linewidth=2)
# plt.xticks(size=15)
# plt.yticks([0, 0.25, 0.5, 0.75, 1], size=15)
# plt.title('Expected Loss', size=15)
# plt.ylabel('Probability', size=15)
# plt.ylim((0,1))
# plt.grid()
# plt.show()




#partie 3 estimation de CDF (cumulative distribution function) en utilisant QAE
print("----------estimaton de CDF-------------------------")

# fixer la valeur x pour estimer le CDF
x_eval = 2

# définition du circuit de comparaison
cdf_objective = FixedValueComparator(agg.num_sum_qubits, x_eval+1, geq=False)

# définition du "Multivariate Problem"
multivariate_cdf = MultivariateProblem(u, agg, cdf_objective)

# obtenir le nombre de qubits nécessaires pour représenter le problème
num_qubits = multivariate_cdf.num_target_qubits
# obtenir le nombre de qubits auxiliaires nécessaires pour représenter le problème
num_ancillas = multivariate_cdf.required_ancillas()  

# construire le circuit
q = QuantumRegister(num_qubits, name='q')
q_a = QuantumRegister(num_ancillas, name='q_a')
qc = QuantumCircuit(q, q_a)
multivariate_cdf.build(qc, q, q_a)


# exécuter l'estimation
num_eval_qubits = 4
ae_cdf = AmplitudeEstimation(num_eval_qubits, multivariate_cdf)
result_cdf = ae_cdf.run(quantum_instance=BasicAer.get_backend('statevector_simulator'))


# print results
# print(result_cdf)
print('Exact value:    \t%.4f' % cdf[x_eval])
print('Estimated value:\t%.4f' % result_cdf['estimation'])
print('Probability:    \t%.4f' % result_cdf['max_probability'])


#partie 4 estimation de VaR
print("---------- estimation de VaR----------------------")

#fonction auxiliaire utilisée pour l'estimation de la VaR. 
#Cette fonction effectue une stimulation PDF en utilisant x comme "valeur à comparer".
def EstimatedVaR(x):


	cdf_objective = FixedValueComparator(agg.num_sum_qubits, x+1, geq=False)


	multivariate_var = MultivariateProblem(u, agg, cdf_objective)

	num_eval_qubits = 4
	ae_var = AmplitudeEstimation(num_eval_qubits, multivariate_var)
	result_var = ae_var.run(quantum_instance=BasicAer.get_backend('statevector_simulator'))
	    
	#print("result_var['estimation']: ", result_var['estimation'])
	return result_var['estimation']



objective = lambda x: EstimatedVaR(x)

bisection_result = bisection_search(objective, 1-alpha, min(losses) - 1, max(losses), low_value=0, high_value=1)

var = bisection_result['level']

print('Estimated Value at Risk: %2d' % var)
print('Exact Value at Risk:     %2d' % exact_var)
print('Estimated Probability:    %.3f' % bisection_result['value'])
print('Exact Probability:        %.3f' % cdf[exact_var])


#partie 5 estimation de CVaR
print("---------- estimation de CVaR --------------------")

breakpoints = [0, var]
slopes = [0, 1]
offsets = [0, 0]  
f_min = 0
f_max = 3 - var
c_approx = 0.25

cvar_objective = UnivariatePiecewiseLinearObjective(
    agg.num_sum_qubits,
    0,
    2**agg.num_sum_qubits-1,  
    breakpoints, 
    slopes, 
    offsets, 
    f_min, 
    f_max, 
    c_approx
)

var = 2
#définition du multivariate problem
multivariate_cvar = MultivariateProblem(u, agg, cvar_objective)

# obtenir le nombre de qubits nécessaires pour représenter le problème
num_qubits = multivariate_cvar.num_target_qubits
# obtenir le nombre de qubits auxiliaires nécessaires pour représenter le problème
num_ancillas = multivariate_cvar.required_ancillas()

#construire le circuit
q = QuantumRegister(num_qubits, name='q')
q_a = QuantumRegister(num_ancillas, name='q_a')
qc = QuantumCircuit(q, q_a)
multivariate_cvar.build(qc, q, q_a)


# exécuter l'estimation
num_eval_qubits = 7
ae_cvar = AmplitudeEstimation(num_eval_qubits, multivariate_cvar)
result_cvar = ae_cvar.run(quantum_instance=BasicAer.get_backend('statevector_simulator'))

#Calcul de CVaR
cvar = (result_cvar['estimation'] / (1.0 - bisection_result['value']) + var)

#print results
print('Exact CVaR:    \t%.4f' % exact_cvar)
print('Estimated CVaR:\t%.4f' % cvar)
print('Probability:   \t%.4f' % result_cvar['max_probability'])


