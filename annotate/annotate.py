import os
import ast
import inspect
import importlib

class AnnotationTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def visit_Assign(self, node):
        # Helper function to determine if a node represents a call to `fn.` and is not `fn.external_source`
        def is_target_fn_call(node):
            # Traverse up the AST from the current node to check if this call is part of the 'fn' namespace
            def is_fn_namespace(node):
                # If the node is an attribute access (e.g., fn.decoders or fn.decoders.image), keep going up
                if isinstance(node, ast.Attribute):
                    # Can't wrap random functions under conditionals
                    if (node.attr == 'random'):
                        return False
                    return is_fn_namespace(node.value)
                # If the node is the 'fn' namespace, return True
                elif isinstance(node, ast.Name) and node.id == 'fn':
                    return True
                # If we reach here, it's not part of the 'fn' namespace
                return False

            # Check if the node is a call and the function being called belongs to the 'fn' namespace
            if isinstance(node, ast.Call) and is_fn_namespace(node.func):
                # You could add more checks here if you need to exclude specific functions
                # For example, to exclude 'fn.external_source', you could check the last attribute:
                # print(node.func.attr if isinstance(node.func, ast.Attribute) else "pass")
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'external_source':
                    return False
                return True
            return False

        # Helper function to check for `fn.` call involvement, directly or in operations
        def involves_target_fn_call(node):
            if isinstance(node, ast.BinOp):
                # Check both sides of the binary operation
                return involves_target_fn_call(node.left) or involves_target_fn_call(node.right)
            else:
                return is_target_fn_call(node)

        # Apply conditional wrapping
        if involves_target_fn_call(node.value):
            self.counter += 1  # Increment counter for unique condition checks
            test_condition = self._build_test_condition()

            # Construct conditional assignment
            conditional_node = ast.If(
                test=test_condition,
                body=[ast.copy_location(ast.Assign(targets=node.targets, value=node.value), node)],
                orelse=[]
            )
            return ast.copy_location(conditional_node, node)

        return node

    def _build_test_condition(self):
        # Helper method to build the conditional test with `self.counter`
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr='condition',
                ctx=ast.Load()
            ),
            args=[
                ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='profile_helpers', ctx=ast.Load()),
                ast.Num(n=self.counter)
            ],
            keywords=[]
        )

def get_annotated_computational_graph(pipe):
    class_source = inspect.getsource(pipe)

    # Parse the class source to get the AST (Abstract Syntax Tree)
    class_ast = ast.parse(class_source)
    function_source = None
    for item in class_ast.body:
        if isinstance(item, ast.ClassDef):
            for node in item.body:
                if isinstance(node, ast.FunctionDef) and node.name == 'define_graph':
                    function_source = node
                    break

    # Apply the transformation and convert the modified AST back to source
    transformer = AnnotationTransformer()
    new_func = transformer.visit(function_source)
    for i, item in enumerate(class_ast.body):
        if isinstance(item, ast.ClassDef):
            for j, node in enumerate(item.body):
                if isinstance(node, ast.FunctionDef) and node.name == 'define_graph':
                    # Print the AST of the 'define_graph' function
                    item.body[j] = new_func
                    break

    new_class = ast.unparse(class_ast)
    new_class = f"from pipeline import BasePipeline\nimport nvidia.dali.fn as fn\nimport nvidia.dali.types as types\n{new_class}"
    with open(f'{os.getcwd()}/__temp.py', 'w') as file:
        file.writelines(new_class)
    module = importlib.import_module('__temp')
    class_obj = getattr(module, pipe.__name__, None)
    return class_obj
