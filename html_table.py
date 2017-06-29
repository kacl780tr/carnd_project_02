class HTMLTable(list):
    """
    Override list to render HTML table in those silly Ipython notebooks
    """
    def _repr_html_(self):
        html = ["<table>"]
        for rw in self:
            html.append("<tr>")
            for cl in rw:
                html.append("<td>{0}</td>".format(cl))
            html.append("</tr>")
        html.append("</table>")
        return "".join(html)
