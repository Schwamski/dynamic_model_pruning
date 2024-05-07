import gymnasium as gym
import neat
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager

CONFIG_PATH = 'bipedal-neat-config.txt'
CONFIG = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     CONFIG_PATH)
CHECKPOINT = 0
PERFORMANCE_THRESHOLD = 250
NODE_PENALTY = 0.1

env = gym.make('BipedalWalker-v3', render_mode='rgb_array')

def bipedal_psi(action): 
    """ 
    Just clips the action to be -1 to 1 since each joint in bipedal walker requires this domain
    """
    return np.clip(action, -1, 1)

def eval_genome(args, num_penalty, gamma = 1.0):
    """ 
    Runs an entire game loop for this given genome and config to evaluate the final score and penalized node fitness
    """
    genome_id, genome, config = args
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    observation, _ = env.reset()
    total_reward = 0.0
    discount = 1.0 
    done = False
    while not done:
        inputs = observation
        action = bipedal_psi(net.activate(inputs))
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward * discount
        discount *= gamma 
        done = terminated or truncated

    env.close()
    num_nodes = len(genome.nodes)   
    node_penalty = num_nodes * NODE_PENALTY * num_penalty.value
    return genome_id, total_reward, node_penalty

def eval_genomes(genomes, config, num_penalty):
    """
    Does the eval genome but over all genomes and IN PARALLEL
     -- Also updates the penalty in a way to keep it persistant once thresholds have been reached
    
    """
    with Pool(cpu_count()) as pool:
        total = len(genomes)
        pbar = tqdm(total=total, desc="Processing Genomes", leave=False)
        
        for _, genome in genomes:
            genome.fitness = 0
            
            
        # Makes this algo parallel over all the GENOMES BB
        jobs = [(genome_id, genome, config) for genome_id, genome in genomes]
        results = [pool.apply_async(eval_genome, (job, num_penalty), callback=lambda res: update(res, genomes, pbar)) for job in jobs]
        pool.close()
        pool.join()
        pbar.close()

        max_total_reward = 0
        for result in results:
            genome_id, total_reward, node_penalty = result.get()
            max_total_reward = max(max_total_reward, total_reward)
            fitness = total_reward - node_penalty
            for gid, genome in genomes:
                if gid == genome_id:
                    genome.fitness = fitness
                    break
        
        # Actually granularly increment the penalties 
        if max_total_reward >= PERFORMANCE_THRESHOLD: 
            num_penalty.value += 1
            print(f"Num Penalty increased to {num_penalty.value}")

def update(result, genomes, pbar):
    """ 
    Given the calculated rewards and penalties actually sets the fitness of the correct gene. 
    TODO: Could be made more efficient 
    """
    genome_id, total_reward, node_penalty = result
    fitness = total_reward - node_penalty
    for gid, genome in genomes:
        if gid == genome_id:
            genome.fitness = fitness
            break
    pbar.update(1)

def main():
    """ 
    Main loop where you run neat code, nuff said
    """
    manager = Manager()
    num_penalty = manager.Value('d', 0)
    
    # Checkpointing each of the generations so you can load it back from whenever and keep training
    if CHECKPOINT > 0: 
        checkpoint_path = f"checkpoints/bipedal standard/neat-checkpoint-{CHECKPOINT}"
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)
    else:  
        pop = neat.Population(CONFIG)
    
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    checkpointer = neat.Checkpointer(1, filename_prefix="checkpoints/bipedal standard/neat-checkpoint-")
    checkpointer.current_generation = CHECKPOINT + 1
    checkpointer.last_generation_checkpoint = CHECKPOINT
    pop.add_reporter(checkpointer)

    winner = pop.run(lambda genomes, config: eval_genomes(genomes, config, num_penalty), 300)
    print("Winner:", winner)

if __name__ == '__main__':
    main()
