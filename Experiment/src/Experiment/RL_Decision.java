package Experiment;

import jsc.distributions.Bernoulli;

import java.util.*;
import java.util.concurrent.*;

/* The Reinforcement Learning module makes online decision for the task-worker selection
 * Created by Zehong on 3/17/2017 0017.
 */
abstract class RL_Decision {
    // System time
    int m_t;
    // Available action list
    List<Action> m_available_action;
    // Multi-Thread service
    ThreadPoolExecutor m_thread_service;
    // Vector of State (Used for Multi-Threading)
    DVState m_state_vec;

    // Initialize the inner state
    RL_Decision(int t0, State S0)
    {
        m_t = t0;
        m_available_action = S0.getAvailableAction();
        int nThreads = Runtime.getRuntime().availableProcessors()/2+1;
        m_thread_service = new ThreadPoolExecutor(nThreads, nThreads,
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<Runnable>(),
                new WorkerThreadFactory());
        m_state_vec = new DVState(S0, nThreads);
    }

    // Make the task-worker decision
    abstract Action getDecision(State St, Prob_Model model, Obj_Function obj) throws InterruptedException;

    // Get the system time
    int getT()
    {
        return m_t;
    }

    // Update the state
    void UpdateSate(Action a, double label)
    {
        m_state_vec.setEntry(a.i, a.j, label);
    }

    // Close the multi-threading service
    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        m_thread_service.shutdown();
    }
}

class WorkerThreadFactory implements ThreadFactory {
    private int counter = 0;

    public Thread newThread(Runnable r) {
        return new Thread(r, Integer.toString(counter++));
    }
}



/* The random selection algorithm always uniformly select the action.
 */
class Random_RL extends RL_Decision {

    Random_RL(int t0, State S0)
    {
        super(t0, S0);
    }

    Action getDecision(State St, Prob_Model model, Obj_Function obj)
    {
        // Uniformly select the action
        int index_of_action = ThreadLocalRandom.current().nextInt(m_available_action.size());
        // Remove the action from the action set
        Action at = m_available_action.remove(index_of_action);
        // The time plus one
        m_t++;

        return at;
    }
}

/* The optimistic knowledge gradient method
 */
class OptGrad_RL extends RL_Decision {
    OptGrad_RL(int t0, State S0)
    {
        super(t0, S0);
    }

    Action getDecision(State St, Prob_Model model, Obj_Function obj) throws InterruptedException {
        // Build the root node
        CrowdNode root = new CrowdNode(null, null, 0, obj, true, 1.0);
        // Give the model
        root.crowdModel = model;
        root.objValue = 0;
        // Build the single level tree to identify the best
        root.Gene_Children(1, m_state_vec, m_available_action, m_thread_service);
        // Find the best parent action
        List<CrowdNode> nodes = root.childNodes.get(0);
        Action at = nodes.get(0).parentAction;
        // The time plus one
        m_t++;

        // Remove the action
        int index = m_available_action.indexOf(at);
        m_available_action.remove(index);

        return at;
    }
}

/* The epsilon greedy method
 */
class EpsGrad_RL extends RL_Decision {
    private final static double epsilon = 0.99;

    EpsGrad_RL(int t0, State S0)
    {
        super(t0, S0);
    }

    Action getDecision(State St, Prob_Model model, Obj_Function obj) throws InterruptedException {
        Bernoulli random_ob = new Bernoulli(epsilon);
        double exploitation_or_exploration = random_ob.random();
        // Exploitation or exploration
        int index_of_action;
        Action at;
        if (exploitation_or_exploration<0.5)
        {
            // exploration by uniformly selecting the action
            index_of_action = ThreadLocalRandom.current().nextInt(m_available_action.size());
            at = m_available_action.remove(index_of_action);
        }
        else
        {
            // exploit the current estimate by selecting the best action
            CrowdTree tree = new CrowdTree(m_state_vec, model, obj, m_available_action, m_thread_service, true);
            at = tree.root.Find_Best_Action();
            index_of_action = m_available_action.indexOf(at);
            m_available_action.remove(index_of_action);
        }
        // The time plus one
        m_t++;

        return at;
    }
}








