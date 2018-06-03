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
        protected BackendFunction(
            IEnumerable<ExecutableModelNode> inputs, 
            IEnumerable<ExecutableModelNode> outputs,
            IEnumerable<ExecutableModelNode> updates)
        {
            Outputs = outputs;
            Inputs = inputs;
            Updates = updates;
        }

        /// <summary>
        /// Gets the outputs to fetch as part of the function
        /// </summary>
        protected IEnumerable<ExecutableModelNode> Outputs { get; }

        /// <summary>
        /// Gets the default inputs for the function
        /// </summary>
        protected IEnumerable<ExecutableModelNode> Inputs { get; }

        /// <summary>
        /// Gets the updates to perform as part of the function
        /// </summary>
        protected IEnumerable<ExecutableModelNode> Updates { get; }

        /// <summary>
        /// Executes the function against the backend
        /// </summary>
        /// <param name="inputs">Input values for the function. Values provided here override the defaults
        /// provided in the <see cref="Inputs"/> property.</param>
        /// <returns>Returns the values fetched for the specified outputs</returns>
        public abstract IEnumerable<object> Execute(IEnumerable<object> inputs);

    }
}
