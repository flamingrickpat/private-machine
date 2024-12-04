from pm.grapheditor.decorator import global_pydantic_classes, global_func_classes, global_enum_classes, global_pure_func_classes, global_data_models

def get_cls(class_name: str):
    cls = None
    if class_name in globals():
        return globals()[class_name]

    if class_name in global_pydantic_classes.keys():
        cls = global_pydantic_classes[class_name]
    if class_name in global_func_classes.keys():
        cls = global_func_classes[class_name]
    if class_name in global_pure_func_classes.keys():
        cls = global_pure_func_classes[class_name]
    if class_name in global_enum_classes.keys():
        cls = global_enum_classes[class_name]
    if class_name in global_data_models.keys():
        cls = global_data_models[class_name]
    if class_name == "ToolNodeDataModel":
        from pm.grapheditor.type_tool import ToolJsonNodeDataModel
        cls = ToolJsonNodeDataModel
    return cls

