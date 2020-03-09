#!/bin/sh
#SBATCH -p gpu 					# Cola a la quie se va a realizar la conexión
#SBATCH --account=gpu				# Cuenta de Slurm (Requerida)
#SBATCH --gres=gpu:1				# GPUs solicitadas
#SBATCH -N 1					# Nodos Requeridos
#SBATCH -n 16					# Cores por nodo
#SBATCH --mem=32G				# Tamaño Memoria RAM
#SBATCH -t 02:00:00				# Tiempo de uso [IMPORTANTE]
#SBATCH --mail-user=cy.andrade@uniandes.edu.co 	# Correo del usuario
#SBATCH --mail-type=ALL				# Tipo del email
#SBATCH --job-name=gan_128x128_50_it	# Nombre del Job
#SBATCH -o ./Job_logs/gan_128*128_50_it.txt

module load tensorflow/GPU.2.0.0
module load anaconda/python3.7
#python script_cvae.py prueba1
#python script_gan_cvae.py prueba2
python script_gan.py 256x256_50_it
