using Neuromatic.Core;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// Visitor used to compile the model to an executable graph
    /// </summary>
    public abstract class ModelBackend: IDisposable
    {
        /// <summary>
        /// Visits an input layer
        /// </summary>
        /// <param name="layer">Input layer to visit</param>
        public abstract ExecutableModelNode Placeholder(string name, long[] shape);

        /// <summary>
        /// Gets the output for the visitor
        /// </summary>
        public abstract ExecutableModel Output { get; }

        /// <summary>
        /// Locates a node in the graph that is managed by the backend
        /// </summary>
        /// <param name="name">Name of the model node</param>
        /// <returns>Returns the found model node. Returns null when the node could not be found.</returns>
        public abstract ExecutableModelNode Node(string name);
        
        #region IDisposable Support
        
        /// <summary>
        /// Disposes any resources used by the backend
        /// </summary>
        /// <param name="disposing"></param>
        protected virtual void Dispose(bool disposing)
        {
            
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            Dispose(true);
        }

        #endregion
    }
}
