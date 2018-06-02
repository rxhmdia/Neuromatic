using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// Defines an executable function that can be executed against a backend.
    /// This is typically used by the framework to execute optimizer functions or other backend logic.
    /// </summary>
    public abstract class BackendFunction
    {
        /// <summary>
        /// Initializes a new instance of <see cref="BackendFunction"/>
        /// </summary>
        /// <param name="inputs">Mapping for inputs with their default values</param>
        /// <param name="outputs">List of outputs to fetch as part of the function</param>
        protected BackendFunction(IDictionary<ExecutableModelNode, object> inputs, IEnumerable<ExecutableModelNode> outputs)
        {
            Outputs = outputs;
            Inputs = inputs;
        }

        /// <summary>
        /// Gets the outputs to fetch as part of the function
        /// </summary>
        protected IEnumerable<ExecutableModelNode> Outputs { get; }

        /// <summary>
        /// Gets the default inputs for the function
        /// </summary>
        protected IDictionary<ExecutableModelNode, object> Inputs { get; }

        /// <summary>
        /// Executes the function against the backend
        /// </summary>
        /// <param name="inputs">Input values for the function. Values provided here override the defaults
        /// provided in the <see cref="Inputs"/> property.</param>
        /// <returns>Returns the values fetched for the specified outputs</returns>
        public abstract IEnumerable<object> Execute(IDictionary<ExecutableModelNode, object> inputs);

        /// <summary>
        /// Executes the function against the backend
        /// </summary>
        /// <returns>Returns the values fetched for the specified outputs</returns>
        public IEnumerable<object> Execute()
        {
            return Execute(new Dictionary<ExecutableModelNode, object>());
        }

        /// <summary>
        /// Merges the provided input override values with the defaults provided in the <see cref="Inputs"/> property.
        /// The defaults are always overwritten by the provided input overrides.
        /// </summary>
        /// <param name="inputOverrides">Set of input override values</param>
        /// <returns>The merged input values for the function</returns>
        protected IDictionary<ExecutableModelNode, object> MergeInputValues(IDictionary<ExecutableModelNode, object> inputOverrides)
        {
            var output = new Dictionary<ExecutableModelNode, object>();

            foreach(var keyValuePair in Inputs)
            {
                if(!inputOverrides.ContainsKey(keyValuePair.Key))
                {
                    output.Add(keyValuePair.Key, keyValuePair.Value);
                }
            }

            foreach(var keyValuePair in inputOverrides)
            {
                output.Add(keyValuePair.Key, keyValuePair.Value);
            }

            return output;
        }
    }
}
