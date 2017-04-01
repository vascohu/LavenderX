package Experiment;

import org.apache.commons.math3.linear.RealVector;

import java.util.*;
import java.util.concurrent.*;

/* Crowd Node represents the node in the crowd tree.
 * Created by Zehong on 3/22/2017 0022.
 */
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

    // Number of Searching Space
    private final static int nSpareSampling = 200;

    // Flag of Sparse Sampling
    public boolean flagSparseSampling;



    // Initialize the Node
    CrowdNode(CrowdNode parent, Action pa_action, int ob_label, Obj_Function obj_function, boolean Optimistic_Or_Not, double label_probability) {
        parentNode = parent;
        parentAction = pa_action;
        observedLabel = ob_label;
        objFun = obj_function;
        flagOpt = Optimistic_Or_Not;
        labelProb = label_probability;
        childNodes = new ArrayList<>();
        flagSparseSampling = (parent != null) && (parent.flagSparseSampling);
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
    private Map<Action, Double> CalActionValue(DVState s_vec, List<Action> available_action_list, ThreadPoolExecutor executor) {
        List<PredictValueTask> futureList = new ArrayList<PredictValueTask>();
        Map<Action, Double> action_value = new HashMap<>();
        List<Action> actions = new ArrayList<>();
        if(flagSparseSampling && (nSpareSampling<available_action_list.size())) {
            List<Action> ss_aal = new ArrayList<>(available_action_list);
            for (int i = 0; i < nSpareSampling; ++i) {
                int index_of_action = ThreadLocalRandom.current().nextInt(ss_aal.size());
                PredictValueTask myTask = new PredictValueTask(s_vec, ss_aal.get(index_of_action), crowdModel, objFun, flagOpt);
                futureList.add(myTask);
                actions.add(ss_aal.remove(index_of_action));
            }
        }
        else
        {
            for(Action a: available_action_list){
                PredictValueTask myTask = new PredictValueTask(s_vec, a, crowdModel, objFun, flagOpt);
                futureList.add(myTask);
                actions.add(a);
            }
        }
        try {
            List<Future<Double>> futures = executor.invokeAll(futureList);
            Iterator<Future<Double>> it1 = futures.iterator();
            Iterator<Action> it2 = actions.iterator();
            while (it1.hasNext()) {
                Future<Double> future = it1.next();
                if(!future.isDone())
                {
                    System.out.println("Error");
                }
                action_value.put(it2.next(), future.get());
            }
        } catch (InterruptedException | NullPointerException | ExecutionException e) {
            System.out.println("Multi-Threading Error: " + e.getMessage());
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
    void Gene_Children(int num_of_children, DVState s_vec, List<Action> available_action_list, ThreadPoolExecutor executor) {

        if(available_action_list.size()>num_of_children)
        {
            Map<Action, Double> action_value = CalActionValue(s_vec, available_action_list, executor);
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
                childNodes.add(Gene_Node(s_vec.get(0), best_action));
                action_value.put(best_action, -1.0);
            }
        }
        else{
            for(Action action:available_action_list)
            {
                childNodes.add(Gene_Node(s_vec.get(0), action));
            }
        }
    }

    Action Find_Best_Action()
    {
        return childNodes.get(0).get(0).parentAction;
    }
}


class PredictValueTask implements Callable<Double> {
    private DVState m_s_vec;
    private Action m_a;
    private Obj_Function m_obj_fun;
    private Prob_Model m_model;
    private boolean m_flag_opt;

    PredictValueTask(DVState s_vec, Action a, Prob_Model model, Obj_Function obj_fun, boolean flag_opt) {
        m_s_vec = s_vec;
        m_a = a;
        m_model = model;
        m_obj_fun = obj_fun;
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

    private double Cal_Action_Value()
    {
        RealVector label_prob = m_model.getLabelProb(m_a);
        double m_obj = 0;
        int index_of_state = Integer.parseInt(Thread.currentThread().getName());
        State s = m_s_vec.get(index_of_state+1);
        for(int i=0; i<label_prob.getDimension(); ++i) {
            m_obj += label_prob.getEntry(i) * Cal_Label_Value(s, i + 1);
        }
        return m_obj;
    }

    private double Cal_Opt_Action_value()
    {
        double m_obj = 0;
        int index_of_state = 1;//Integer.parseInt(Thread.currentThread().getName());
        State s = m_s_vec.get(index_of_state+1);
        for(int i=0; i<m_model.getClass_Num(); ++i) {
            double value = Cal_Label_Value(s, i + 1);
            if(value>m_obj)
            {
                m_obj = value;
            }
        }
        return m_obj;
    }

    @Override
    public Double call()
    {
        if(m_flag_opt)
        {
            return Cal_Opt_Action_value();
        }
        else
        {
            return Cal_Action_Value();
        }
    }
}