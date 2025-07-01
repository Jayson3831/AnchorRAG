SPARQL_TEMPLATES = {
    'name2id': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?entity WHERE {
            ?entity ns:type.object.name ?name .
            FILTER(?name = "%s"@en)
        }
    """,
    
    'relations': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?relation
        WHERE {
            {
                ns:%s ?relation ?x .
            }
            UNION
            {
                ?x ?relation ns:%s .
            }
        }
    """,

    'connected_entities': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?connectedEntity
        WHERE {
            {
                ns:%s ns:%s ?connectedEntity .
            }
            UNION
            {
                ?connectedEntity ns:%s ns:%s .
            }
        }
    """,

    'head_relations': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?relation
        WHERE { ns:%s ?relation ?x . }
    """,
    
    'tail_relations': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?relation
        WHERE { ?x ?relation ns:%s . }
    """,
    
    'tail_entities': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?tailEntity
        WHERE { ns:%s ns:%s ?tailEntity . }
    """,
    
    'head_entities': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?tailEntity
        WHERE { ?tailEntity ns:%s ns:%s . }
    """,
    
    'entity_info': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?tailEntity
        WHERE {
            { ?entity ns:type.object.name ?tailEntity . FILTER(?entity = ns:%s) }
            UNION
            { ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity . FILTER(?entity = ns:%s) }
        }
    """,
    
    'sparql_id': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT DISTINCT ?tailEntity
        WHERE {
            VALUES ?entity { ns:%s }
            {
                ?entity ns:type.object.name ?tailEntity .
            }
            UNION
            {
                ?entity owl:sameAs ?tailEntity .
            }
        }
    """,
    
    'sparql_query': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        ASK {
            ?entity ns:type.object.name "%s"@en .
        }
    """,
    
    'name_head_relations': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?relation
        WHERE {
            ?entity ?relation ?x .
            ?entity ns:type.object.name "%s"@en .
        }
    """,
    
    'name_tail_relations': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?relation
        WHERE {
            ?x ?relation ?entity .
            ?entity ns:type.object.name "%s"@en .
        }
    """,
    
    'name_tail_entities': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?tailEntity
        WHERE {
            ?entity ns:%s ?tailEntity .
            ?entity ns:type.object.name "%s"@en .
        }
    """,
    
    'name_head_entities': """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?tailEntity
        WHERE {
            ?tailEntity ns:%s ?entity .
            ?entity ns:type.object.name "%s"@en .
        }
    """
}