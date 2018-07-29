using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorFlow;

namespace Neuromatic.Optimizers
{
    /// <summary>
    /// Defines an optimization function to be run when training a neural network
    /// </summary>
    public abstract class Optimizer
    {
        /// <summary>
        /// Gets or sets the operations that the optimizer needs 
        /// to execute for each training batch
        /// </summary>
        protected IEnumerable<TFOperation> Operations { get;set;}
        
        /// <summary>
        /// Compiles the optimizer
        /// </summary>
        /// <param name="context">Context to use for compilation</param>
        /// <param name="loss">Loss function to use</param>
        /// <param name="parameters">Parameters to optimize</param>
        public abstract void Compile(ModelCompilationContext context, TFOutput loss, IEnumerable<TFOutput> parameters);

        /// <summary>
        /// Executes the optimizer function for a set of inputs
        /// </summary>
        /// <param name="session"></param>
        /// <param name="inputs"></param>
        public void Execute(TFSession session, Dictionary<TFOutput, Array> inputs)
        {
            var runner = session.GetRunner();

            foreach(var keyValuePair in inputs)
            {
                runner.AddInput(keyValuePair.Key, new TFTensor(keyValuePair.Value));
            }

            runner.AddTarget(Operations.ToArray());
            runner.Run();
        }
    }
}
