package Experiment;

import jsc.distributions.Bernoulli;
import org.apache.commons.math3.linear.RealVector;

import java.util.*;
import java.util.concurrent.*;

/* The Reinforcement Learning module makes online decision for the task-worker selection
 * Created by Zehong on 3/17/2017 0017.
 */
abstract class RL_Decision {
    List<Action> m_available_action;
    int m_t;

    // Initialize the inner state
    RL_Decision(int t0, State S0)
    {
        m_t = t0;
        m_available_action = S0.getAvailableAction();
    }

    // Make the task-worker decision
    abstract Action getDecision(State St, Prob_Model model, Obj_Function obj) throws InterruptedException;

    // Get the system time
    int getT()
    {
        return m_t;
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
        root.Gene_Children(1,St, m_available_action);
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
    private final double epsilon = 0.99;

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
            CrowdTree tree = new CrowdTree(St, model, obj, m_available_action, true);
            tree.GeneTree();
            at = tree.root.Find_Best_Action();
            index_of_action = m_available_action.indexOf(at);
            m_available_action.remove(index_of_action);
        }
        // The time plus one
        m_t++;

        return at;
    }
}


/* The tree approximates the value function with a d-depth tree.
 * We build the tree with depth-first search.
 */
class CrowdTree {
    // The depth of the tree
    public int depth = 5;
    // The width of the sparse tree
    final int width = 5;
    // The discount of RL
    final double gamma = 0.9;
    // The state
    private final State m_state;
    // The root
    public final CrowdNode root;
    // The available action set
    private List<Action> m_available_action;

    CrowdTree(State St, Prob_Model model, Obj_Function obj, List<Action> available_action)
    {
        m_state = St;
        root = new CrowdNode(null, null, 0, obj, false, 1.0);
        root.crowdModel = model;
        root.objValue = obj.getObjValue(model);
        m_available_action = available_action;
        if(m_available_action.size()<depth)
        {
            depth = m_available_action.size();
        }
    }

    CrowdTree(State St, Prob_Model model, Obj_Function obj, List<Action> available_action, boolean Sparse_Or_Not)
    {
        m_state = St;
        root = new CrowdNode(null, null, 0, obj, false, 1.0, Sparse_Or_Not);
        root.crowdModel = model;
        root.objValue = obj.getObjValue(model);
        m_available_action = available_action;
        if(m_available_action.size()<depth)
        {
            depth = m_available_action.size();
        }
    }

    private void TreeGrow(CrowdNode p_node, int d) throws InterruptedException {
        if(d!=0)
        {
            p_node.Gene_Children(width, m_state, m_available_action);
            double action_value = -Double.MAX_VALUE;
            for(List<CrowdNode> action_node: p_node.childNodes)
            {
                Action parent_action = action_node.get(0).parentAction;
                int index_of_action = m_available_action.indexOf(parent_action);
                m_available_action.remove(index_of_action);
                RealVector label_prob = p_node.crowdModel.getLabelProb(parent_action);
                double this_action_value = 0;
                for(CrowdNode node: action_node)
                {
                    m_state.setEntry(parent_action.i, parent_action.j, node.observedLabel);
                    TreeGrow(node, d-1);
                    this_action_value += label_prob.getEntry(node.observedLabel-1)*node.objValue;
                }
                m_state.setEntry(parent_action.i, parent_action.j, 0);
                m_available_action.add(index_of_action, parent_action);
                if (this_action_value>action_value)
                {
                    action_value = this_action_value;
                }
            }
            p_node.objValue += gamma*(action_value - p_node.objValue);
        }
    }

    // Generate the tree
    void GeneTree() throws InterruptedException {
        TreeGrow(root, depth);
    }
}



class CrowdNode {
    // Tree Structure
    public CrowdNode parentNode;
    public List<List<CrowdNode>> childNodes; // Every action may generate classNum states.

    // Node Data
    public Prob_Model crowdModel;
    public double objValue;
    public Obj_Function objFun;
    public boolean flagOpt;

    // Node Generation
    public Action parentAction;
    public int observedLabel;
    public double labelProb;

    // Number of Threads
    public final int nThreads = 6;//Runtime.getRuntime().availableProcessors()/2;
    public final int nSpareSampling = 100;

    // Flag of Sparse Sampling
    public boolean flagSparseSampling;



    // Initialize the Node
    CrowdNode(CrowdNode parent, Action pa_action, int ob_label, Obj_Function obj_function, boolean Optimistic_Or_Not, double label_probability)
    {
        parentNode = parent;
        parentAction = pa_action;
        observedLabel = ob_label;
        objFun = obj_function;
        flagOpt = Optimistic_Or_Not;
        labelProb = label_probability;
        childNodes = new ArrayList<>();
        if(parent==null)
        {
            flagSparseSampling = false;
        }
        else
        {
            flagSparseSampling = parent.flagSparseSampling;
        }
        //System.out.println(nThreads);
    }

    // Initialize the Node
    CrowdNode(CrowdNode parent, Action pa_action, int ob_label, Obj_Function obj_function,
              boolean Optimistic_Or_Not, double label_probability, boolean Sparse_Or_Not)
    {
        parentNode = parent;
        parentAction = pa_action;
        observedLabel = ob_label;
        objFun = obj_function;
        flagOpt = Optimistic_Or_Not;
        labelProb = label_probability;
        childNodes = new ArrayList<>();
        flagSparseSampling = Sparse_Or_Not;
    }


    // Calculate the action-value table
    private Map<Action, Double> CalActionValue(State state, List<Action> available_action_list) throws InterruptedException {
        WorkerThreadFactory WT = new WorkerThreadFactory();
        BlockingQueue<Runnable> BQ = new LinkedBlockingQueue<>();
        ThreadPoolExecutor executor = new ThreadPoolExecutor(nThreads, nThreads, 0L,
                TimeUnit.MILLISECONDS, BQ, WT);
        Vector<State> s_vec = new Vector<>(nThreads+1);
        s_vec.add(0,state);
        for(int i=1; i<nThreads+1; ++i)
        {
            s_vec.add(i, null);
        }
        Vector<PredictValueTask> Tasks = new Vector<>();
        Vector<Action> actions = new Vector<>();
        if(flagSparseSampling && (nSpareSampling<available_action_list.size()))
        {
            List<Action> ss_aal = new ArrayList<>(available_action_list);
            for(int i=0; i<nSpareSampling; ++i)
            {
                int index_of_action = ThreadLocalRandom.current().nextInt(ss_aal.size());
                PredictValueTask myTask = new PredictValueTask(s_vec, ss_aal.get(index_of_action), crowdModel, objFun, flagOpt);
                Tasks.add(myTask);
                executor.execute(myTask);
                actions.add(ss_aal.remove(index_of_action));
            }
        }
        else
        {
            for(Action a: available_action_list){
                PredictValueTask myTask = new PredictValueTask(s_vec, a, crowdModel, objFun, flagOpt);
                Tasks.add(myTask);
                executor.execute(myTask);
                actions.add(a);
            }
        }
        executor.shutdown();
        executor.awaitTermination(Integer.MAX_VALUE, TimeUnit.MINUTES);
        Map<Action, Double> action_value = new HashMap<>();
        Iterator<PredictValueTask> task_iterator = Tasks.iterator();
        Iterator<Action> action_iterator = actions.iterator();
        while (task_iterator.hasNext())
        {
            action_value.put(action_iterator.next(), task_iterator.next().getObjValue());
        }
        return action_value;
    }

    private void Cal_Obj_Value(State state)
    {
        state.setEntry(parentAction.i, parentAction.j, observedLabel);
        crowdModel = (parentNode.crowdModel).Copy();
        crowdModel.QuickUpdate(parentAction, state);
        objValue = objFun.getObjValue(crowdModel);
        state.setEntry(parentAction.i, parentAction.j, 0);
    }

    private List<CrowdNode> Gene_Node(State state, Action action)
    {
        List<CrowdNode> new_nodes = new ArrayList<>(crowdModel.getClass_Num());
        RealVector label_prob = crowdModel.getLabelProb(action);
        for(int label = 1; label<=crowdModel.getClass_Num(); ++label)
        {
            CrowdNode node = new CrowdNode(this, action, label, objFun, flagOpt, label_prob.getEntry(label-1));
            node.Cal_Obj_Value(state);
            new_nodes.add(node);
        }
        return new_nodes;
    }

    // Generate the required number of children nodes
    public void Gene_Children(int num_of_children, State state, List<Action> available_action_list) throws InterruptedException {

        if(available_action_list.size()>num_of_children)
        {
            Map<Action, Double> action_value = CalActionValue(state, available_action_list);
            for(int i=0; i<num_of_children; ++i)
            {
                Action best_action = null;
                double value = -1;
                for(Action a:action_value.keySet())
                {
                    if(action_value.get(a)>value+1e-12)
                    {
                        best_action = a;
                        value = action_value.get(a);
                    }
                }
                childNodes.add(Gene_Node(state, best_action));
                action_value.put(best_action, -1.0);
            }
        }
        else{
            for(Action action:available_action_list)
            {
                childNodes.add(Gene_Node(state, action));
            }
        }
    }

    public Action Find_Best_Action()
    {
        return childNodes.get(0).get(0).parentAction;
    }
}


class PredictValueTask implements Runnable {
    private Vector<State> m_s_vec;
    private Action m_a;
    private double m_obj;
    private Obj_Function m_obj_fun;
    private Prob_Model m_model;
    private boolean m_flag_opt;

    PredictValueTask(Vector<State> s_vec, Action a, Prob_Model model, Obj_Function obj_fun, boolean flag_opt) {
        m_s_vec = s_vec;
        m_a = a;
        m_model = model;
        m_obj_fun = obj_fun;
        m_obj = 0;
        m_flag_opt = flag_opt;
    }

    private double Cal_Label_Value(State s, int ob_label)
    {
        s.setEntry(m_a.i, m_a.j, ob_label);
        Prob_Model newModel = m_model.Copy();
        newModel.QuickUpdate(m_a, s);
        double obj = m_obj_fun.getObjValue(newModel);
        s.setEntry(m_a.i, m_a.j, 0);
        return obj;
    }

    private void Cal_Action_Value()
    {
        RealVector label_prob = m_model.getLabelProb(m_a);
        m_obj = 0;
        int index_of_state = Integer.parseInt(Thread.currentThread().getName());
        State s = m_s_vec.elementAt(index_of_state+1);
        if(s == null)
        {
            s = m_s_vec.elementAt(0).copy();
            m_s_vec.set(index_of_state+1, s);
        }
        for(int i=0; i<label_prob.getDimension(); ++i) {
            m_obj += label_prob.getEntry(i) * Cal_Label_Value(s, i + 1);
        }
    }

    private void Cal_Opt_Action_value()
    {
        m_obj = 0;
        int index_of_state = Integer.parseInt(Thread.currentThread().getName());
        State s = m_s_vec.elementAt(index_of_state+1);
        if(s == null)
        {
            s = m_s_vec.elementAt(0).copy();
            m_s_vec.set(index_of_state+1, s);
        }
        for(int i=0; i<m_model.getClass_Num(); ++i) {
            double value = Cal_Label_Value(s, i + 1);
            if(value>m_obj)
            {
                m_obj = value;
            }
        }
    }

    @Override
    public void run()
    {
        if(m_flag_opt)
        {
            Cal_Opt_Action_value();
        }
        else
        {
            Cal_Action_Value();
        }
    }

    double getObjValue()
    {
        return m_obj;
    }
}

class WorkerThreadFactory implements ThreadFactory {
    private int counter = 0;

    public Thread newThread(Runnable r) {
        return new Thread(r, Integer.toString(counter++));
    }
}
